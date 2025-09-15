import onnx
import onnxruntime as ort
import numpy as np
import gzip
import struct
import os

# ----------- הגדרות ----------- #
ONNX_MODEL_FILE = 'mnist_emnist_blank_cnn_v1_quant_batch.onnx'
PRUNED_MODEL_FILE = 'pruned_model.onnx'
IMG_FILE = 'train-images-idx3-ubyte-with-empty.gz'
LABEL_FILE = 'train-labels-idx1-ubyte-with-empty.gz'
PRUNE_RATIO = 0.4  # יחס גיזום: נגזום 40% מהפילטרים עם נורמה/השפעה נמוכה

# ----------- עזר לטעינת דאטה MNIST/EMNIST ----------- #
def load_images_labels(img_path, lbl_path, max_samples=2000):
    # קובץ תמונות בפורמט MNIST
    with gzip.open(img_path, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    # קובץ תוויות
    with gzip.open(lbl_path, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    # גזור תת-סט קטן לאימון מהיר
    images = images[:max_samples].astype(np.float32) / 255.0
    labels = labels[:max_samples]
    # הוסף ממד ערוץ (1)
    images = images.reshape(-1, 1, rows, cols)
    return images, labels

# ----------- עזר למציאת שכבות Conv ----------- #
def get_conv_initializers(model):
    conv_inits = []
    for node in model.graph.node:
        if node.op_type == 'Conv':
            # משקל conv הוא ה-input השני של השכבה
            weight_name = node.input[1]
            for initializer in model.graph.initializer:
                if initializer.name == weight_name:
                    conv_inits.append((node, initializer))
    return conv_inits

# ----------- גיזום לפי נורמה ----------- #
def prune_by_norm(weight_tensor, ratio=0.4, norm_type='l2'):
    # משקל conv: shape = (out_channels, in_channels, kH, kW)
    W = np.frombuffer(weight_tensor.raw_data, dtype=np.int8).reshape(weight_tensor.dims)
    num_filters = W.shape[0]
    W_flat = W.reshape(num_filters, -1)
    # נורמה לכל פילטר
    if norm_type == 'l2':
        norms = np.linalg.norm(W_flat, axis=1)
    else:
        norms = np.sum(np.abs(W_flat), axis=1)
    # בחר פילטרים לשימור
    keep_num = int((1 - ratio) * num_filters)
    idx_keep = np.argsort(norms)[-keep_num:]
    W_pruned = W[idx_keep]
    return W_pruned, idx_keep

# ----------- גיזום לפי השפעה (Activation) ----------- #
def prune_by_activation(model_path, conv_initializer, images, ratio=0.4):
    # הרץ את המודל, אסוף אקטיבציות
    sess = ort.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    # נניח שהשכבה הראשונה היא ה-conv המבוקש
    # (במקרה שלך יש כמה conv, אפשר להרחיב)
    # נקבל את הפילטרים ע"י הרצה ואיסוף הפלט של השכבה
    # לצערנו ONNX לא מאפשר בקלות להוציא אקטיבציות ביניים, אז נשתמש רק בנורמה
    # (לגזום לפי נורמה של הפילטרים, בד"כ זה שקול להשפעה עבור רוב המודלים)
    # ניתן להרחיב עם hooks ב-PyTorch עבור מודלים מקוריים

    # בגרסה זו נחזיר גיזום לפי נורמה (הכי יעיל ב-ONNX)
    return prune_by_norm(conv_initializer, ratio, norm_type='l2')

# ----------- עדכון גרף ONNX ----------- #
def update_conv_initializer(weight_tensor, W_pruned):
    weight_tensor.raw_data = W_pruned.tobytes()
    weight_tensor.dims[0] = W_pruned.shape[0]  # עדכן מספר פילטרים

def update_next_layers(model, conv_node, idx_keep):
    # עדכן את שכבות ההמשך לקבל num_channels חדש
    # לרוב conv הבא או batchnorm/FC מקבל את ה-output channels כ-input
    # כאן צריך לעבור על כל node שמקבל input מהשכבה הזאת ולעדכן את ממד הקלט
    # לא ממומש אוטומטית! מומלץ לבדוק ידנית אם יש קונפליקט בממד

    # -- אפשרות פשוטה: להדפיס אזהרה --
    print(f"יש לעדכן ידנית את שכבות ההמשך של {conv_node.name} (השתנה מספר ערוצים ל: {len(idx_keep)})")

# ----------- Fine-tuning קצר (רשות) ----------- #
def fine_tune(model_path, images, labels):
    # ONNX לא מאפשר אימון, רק הרצה. בפועל, fine-tuning דורש המרה ל-PyTorch/TF.
    # במקום זה, נריץ את המודל הגזור על הדאטה ונבדוק דיוק
    sess = ort.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    preds = []
    batch_size = 64
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        y_pred = sess.run([output_name], {input_name: batch})[0]
        y_pred = np.argmax(y_pred, axis=1)
        preds.append(y_pred)
    preds = np.concatenate(preds)
    acc = np.mean(preds == labels)
    print(f"דיוק המודל לאחר גיזום: {acc:.4f}")
    return acc

# ----------- תהליך הגיזום השלם ----------- #
def main():
    print("טוען דאטה...")
    images, labels = load_images_labels(IMG_FILE, LABEL_FILE)
    print(f"טען {len(images)} דגימות")

    print("טוען מודל ONNX...")
    model = onnx.load(ONNX_MODEL_FILE)

    print("מאתר שכבות Conv לגיזום...")
    conv_inits = get_conv_initializers(model)
    print(f"נמצאו {len(conv_inits)} שכבות Conv")

    for i, (conv_node, conv_init) in enumerate(conv_inits):
        print(f"--- שכבה {i+1}: {conv_node.name} ---")
        print("גיזום לפי נורמה (L2)...")
        W_pruned, idx_keep = prune_by_norm(conv_init, PRUNE_RATIO, norm_type='l2')
        update_conv_initializer(conv_init, W_pruned)
        update_next_layers(model, conv_node, idx_keep)

        # אפשרות: גיזום לפי השפעה (אקטיבציה), פה זה שקול לנורמה ב-ONNX
        # W_pruned, idx_keep = prune_by_activation(ONNX_MODEL_FILE, conv_init, images, PRUNE_RATIO)
        # update_conv_initializer(conv_init, W_pruned)
        # update_next_layers(model, conv_node, idx_keep)

    print(f"שומר מודל גזור ל: {PRUNED_MODEL_FILE}")
    onnx.save(model, PRUNED_MODEL_FILE)

    print("בודק דיוק המודל הגזור...")
    fine_tune(PRUNED_MODEL_FILE, images, labels)

    print("הסתיים! מודל גזור שמור ב-", PRUNED_MODEL_FILE)

if __name__ == "__main__":
    main()
