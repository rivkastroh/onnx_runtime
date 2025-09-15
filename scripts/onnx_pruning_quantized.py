import onnx
import onnxruntime as ort
import numpy as np
import gzip
import struct

ONNX_MODEL_FILE = 'mnist_emnist_blank_cnn_v1_quant_batch.onnx'
PRUNED_MODEL_FILE = 'pruned_model.onnx'
IMG_FILE = 'train-images-idx3-ubyte-with-empty.gz'
LABEL_FILE = 'train-labels-idx1-ubyte-with-empty.gz'
PRUNE_RATIO = 0.4  # יחס גיזום: נגזום 40%

def load_images_labels(img_path, lbl_path, max_samples=2000):
    with gzip.open(img_path, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    with gzip.open(lbl_path, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    images = images[:max_samples].astype(np.float32) / 255.0
    labels = labels[:max_samples]
    images = images.reshape(-1, 1, rows, cols)
    return images, labels

def prune_by_norm(weight_tensor, ratio=0.4):
    W = np.frombuffer(weight_tensor.raw_data, dtype=np.int8).reshape(weight_tensor.dims)
    num_out = W.shape[0]
    W_flat = W.reshape(num_out, -1)
    norms = np.linalg.norm(W_flat, axis=1)
    keep_num = max(1, int((1 - ratio) * num_out))
    idx_keep = np.argsort(norms)[-keep_num:]
    W_pruned = W[idx_keep]
    return W_pruned, idx_keep

def update_initializer(weight_tensor, W_pruned):
    weight_tensor.raw_data = W_pruned.tobytes()
    weight_tensor.dims[0] = W_pruned.shape[0]

def main():
    print("טוען דאטה...")
    images, labels = load_images_labels(IMG_FILE, LABEL_FILE)
    print(f"טען {len(images)} דגימות")
    print("טוען מודל ONNX...")
    model = onnx.load(ONNX_MODEL_FILE)

    pruned_any = False
    # עבור כל node מסוג Conv או Gemm
    for node in model.graph.node:
        if node.op_type in ['Conv', 'Gemm']:
            weight_name = node.input[1]
            # מצא את ה־initializer
            for initializer in model.graph.initializer:
                if initializer.name == weight_name:
                    print(f"גוזם שכבה: {node.op_type}, מזהה: {weight_name}, ממדים: {initializer.dims}")
                    W_pruned, idx_keep = prune_by_norm(initializer, PRUNE_RATIO)
                    update_initializer(initializer, W_pruned)
                    pruned_any = True
                    print(f"נגזמו {initializer.dims[0]-len(idx_keep)} פילטרים/נוירונים")
                    break

    print(f"שומר מודל גזור ל: {PRUNED_MODEL_FILE}")
    onnx.save(model, PRUNED_MODEL_FILE)

    print("בודק דיוק המודל הגזור...")
    sess = ort.InferenceSession(PRUNED_MODEL_FILE)
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
    print("הסתיים! מודל גזור שמור ב-", PRUNED_MODEL_FILE)
    if not pruned_any:
        print("[אזהרה] לא בוצע גיזום! בדוק פורמט המודל או סוג הדאטה.")

if __name__ == "__main__":
    main()