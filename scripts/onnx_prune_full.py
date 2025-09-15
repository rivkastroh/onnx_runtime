import onnx
import numpy as np
from onnx import numpy_helper, shape_inference
import copy
from collections import defaultdict, deque

# ---------- עזר בסיסי ----------

def get_initializer(model, name):
    if not name:
        return None
    for init in model.graph.initializer:
        if init.name == name:
            return init
    return None

def get_node_by_output(model, output_name):
    for node in model.graph.node:
        if output_name in node.output:
            return node
    return None

def get_nodes_by_type(model, op_type):
    return [n for n in model.graph.node if n.op_type == op_type]

def get_attr_i(node, name, default=0):
    if node is None:
        return default
    for a in node.attribute:
        if a.name == name:
            return a.i
    return default

def infer_model(model):
    try:
        return shape_inference.infer_shapes(model)
    except Exception as e:
        print("אזהרה: ONNX shape inference נכשל:", e)
        return model

# ---------- מפות גרף לצרכנים/יצרנים ----------

def build_consumers_map(model):
    consumers = defaultdict(list)  # tensor_name -> [nodes that consume it]
    for node in model.graph.node:
        for inp in node.input:
            if inp:
                consumers[inp].append(node)
    return consumers

# אופרטורים שמעבירים את ממד הערוצים C כמות-שהוא (לא משנים את מספר הערוצים)
CHANNEL_PRESERVING = {
    "Relu", "LeakyRelu", "PRelu", "Sigmoid", "Tanh",
    "BatchNormalization", "Identity", "Cast", "Clip",
    "MaxPool", "AveragePool", "GlobalMaxPool", "GlobalAveragePool",
    # אם יש Add/Concat וכו' צריך טיפול מיוחד; במודל שלך אין.
}

# ---------- גיזום משקלים ----------

def prune_conv_weights(w, prune_ratio):
    # w: [out_channels, in_channels, kH, kW]
    out_ch = w.shape[0]
    num_prune = int(prune_ratio * out_ch)
    if num_prune <= 0 or out_ch - num_prune <= 0:
        keep = np.arange(out_ch, dtype=np.int64)
        return w, keep
    importance = np.sum(np.abs(w), axis=(1, 2, 3))
    keep = np.argsort(-importance)[: out_ch - num_prune].astype(np.int64)
    return w[keep, :, :, :], keep

def prune_conv_in_channels(w, keep_in_idx):
    # w: [out_channels, in_channels, kH, kW]
    return w[:, keep_in_idx, :, :]

def prune_bias(b, keep_out_idx):
    return b[keep_out_idx]

def prune_gemm_in_features(W, keep_feat_idx, transB):
    # transB=1 -> W: [out_features, in_features] -> לגזור בציר 1
    # transB=0 -> W: [in_features, out_features] -> לגזור בציר 0
    if transB == 1:
        return W[:, keep_feat_idx]
    else:
        return W[keep_feat_idx, :]

# ---------- עדכון in_channels ב-Convs בהמשך הזרם ----------

def update_downstream_convs_input_channels(model, start_tensor, keep_out_idx, consumers_map):
    """
    מוצא את כל ה-Convs שמקבלים (ישירות או דרך אופ' ששומרים על C) את ה-tensor 'start_tensor',
    ומעדכן להם את in_channels בהתאם ל-keep_out_idx (פילטרים שנשמרו בשכבה הקודמת).
    נעצרים לפני Conv נוסף – אין טעם לעבור "דרך" Conv, כי הוא כבר יטופל בנקודת הגיזום שלו.
    """
    q = deque([start_tensor])
    visited = set([start_tensor])
    updated_any = False

    while q:
        tname = q.popleft()
        for node in consumers_map.get(tname, []):
            if node.op_type == "Conv":
                w_init = get_initializer(model, node.input[1])
                if w_init is None:
                    continue
                w = numpy_helper.to_array(w_init)
                w_new = prune_conv_in_channels(w, keep_out_idx)
                w_init.CopyFrom(numpy_helper.from_array(w_new, w_init.name))
                updated_any = True
                print(f"עודכן in_channels של Conv {node.name or node.output[0]} ל-{w_new.shape[1]} (בהתאם לגיזום הקודם)")
                # לא דוחפים הלאה אחרי Conv – השרשרת תטופל אחרי הגיזום של ה-Conv הזה עצמו.
            elif node.op_type in CHANNEL_PRESERVING:
                # ממשיכים לעבור קדימה
                for out_t in node.output:
                    if out_t and out_t not in visited:
                        visited.add(out_t)
                        q.append(out_t)
            else:
                # אופ’ אחרים (Flatten/Reshape/Concat/Add/MatMul וכו’) – עצירה
                # במודל שלך ה-Conv הבא מגיע אחרי Relu בלבד, אז זה מכסה את המקרה.
                pass
    return updated_any

# ---------- פונקציית גיזום ראשית ----------

def prune_model(model, prune_ratio=0.4):
    m = copy.deepcopy(model)
    rpt = []

    # שמות פלטי ה-Conv לפי הגרף שלך
    CONV1_OUT = "/features/features.0/Conv_output_0"
    CONV2_OUT = "/features/features.3/Conv_output_0"
    CONV3_OUT = "/features/features.8/Conv_output_0"

    # שמות ה-weight של ה-Gemm-ים לפי ה-initializers בקובץ
    GEMM1_W = "classifier.1.weight"
    GEMM1_B = "classifier.1.bias"
    GEMM2_W = "classifier.5.weight"
    GEMM2_B = "classifier.5.bias"

    consumers_map = build_consumers_map(m)

    # ---- Conv1 ----
    conv1 = get_node_by_output(m, CONV1_OUT)
    if conv1 is None:
        raise RuntimeError(f"לא נמצא Conv1 לפי הפלט {CONV1_OUT}")
    w1_init = get_initializer(m, conv1.input[1])
    b1_init = get_initializer(m, conv1.input[2]) if len(conv1.input) > 2 else None
    w1 = numpy_helper.to_array(w1_init)
    b1 = numpy_helper.to_array(b1_init) if b1_init is not None else None

    w1_p, k1 = prune_conv_weights(w1, prune_ratio)
    w1_init.CopyFrom(numpy_helper.from_array(w1_p, w1_init.name))
    if b1 is not None:
        b1_p = prune_bias(b1, k1)
        b1_init.CopyFrom(numpy_helper.from_array(b1_p, b1_init.name))
    rpt.append({"name": "Conv1", "orig": w1.shape[0], "pruned": w1_p.shape[0]})

    # לעדכן את ה-Convs בהמשך (דרך Relu וכו’) שצורכים את הפלט של Conv1
    update_downstream_convs_input_channels(m, conv1.output[0], k1, consumers_map)

    # ---- Conv2 ----
    conv2 = get_node_by_output(m, CONV2_OUT)
    if conv2 is None:
        raise RuntimeError(f"לא נמצא Conv2 לפי הפלט {CONV2_OUT}")
    w2_init = get_initializer(m, conv2.input[1])
    b2_init = get_initializer(m, conv2.input[2]) if len(conv2.input) > 2 else None
    w2 = numpy_helper.to_array(w2_init)
    b2 = numpy_helper.to_array(b2_init) if b2_init is not None else None

    w2_p, k2 = prune_conv_weights(w2, prune_ratio)
    w2_init.CopyFrom(numpy_helper.from_array(w2_p, w2_init.name))
    if b2 is not None:
        b2_p = prune_bias(b2, k2)
        b2_init.CopyFrom(numpy_helper.from_array(b2_p, b2_init.name))
    rpt.append({"name": "Conv2", "orig": w2.shape[0], "pruned": w2_p.shape[0]})

    # לעדכן את ה-Convs שצורכים את Conv2 (דרך Relu/MaxPool)
    update_downstream_convs_input_channels(m, conv2.output[0], k2, consumers_map)

    # ---- Conv3 ----
    conv3 = get_node_by_output(m, CONV3_OUT)
    if conv3 is None:
        raise RuntimeError(f"לא נמצא Conv3 לפי הפלט {CONV3_OUT}")
    w3_init = get_initializer(m, conv3.input[1])
    b3_init = get_initializer(m, conv3.input[2]) if len(conv3.input) > 2 else None
    w3_before = numpy_helper.to_array(w3_init)  # לשימוש בחישוב H*W של Gemm1
    b3 = numpy_helper.to_array(b3_init) if b3_init is not None else None

    w3_p, k3 = prune_conv_weights(w3_before, prune_ratio)
    w3_init.CopyFrom(numpy_helper.from_array(w3_p, w3_init.name))
    if b3 is not None:
        b3_p = prune_bias(b3, k3)
        b3_init.CopyFrom(numpy_helper.from_array(b3_p, b3_init.name))
    rpt.append({"name": "Conv3", "orig": w3_before.shape[0], "pruned": w3_p.shape[0]})

    # ---- Gemm1 (אחרי Flatten) ----
    # נאתר את צומת ה-Gemm לפי weight/bias הידועים; אם לא – ניקח Gemm ראשון
    gemm_nodes = get_nodes_by_type(m, "Gemm")
    gemm1 = None
    for n in gemm_nodes:
        w_init = get_initializer(m, n.input[1]) if len(n.input) > 1 else None
        if w_init is not None and w_init.name == GEMM1_W:
            gemm1 = n
            break
    if gemm1 is None:
        gemm1 = gemm_nodes[0] if gemm_nodes else None
    if gemm1 is None:
        raise RuntimeError("לא נמצא Gemm1 במודל.")

    transB1 = get_attr_i(gemm1, "transB", 1)  # בדרך כלל ב-PyTorch export transB=1
    W1_init = get_initializer(m, gemm1.input[1])
    B1_init = get_initializer(m, gemm1.input[2]) if len(gemm1.input) > 2 else None
    if W1_init is None:
        raise RuntimeError(f"לא נמצאו משקולות ל-Gemm1 (input[1]={gemm1.input[1]})")
    W1 = numpy_helper.to_array(W1_init)
    # חישוב H*W: in_features / (מס' ערוצים של Conv3 לפני גיזום)
    orig_conv3_out = int(w3_before.shape[0])
    in_feats_len = int(W1.shape[1] if transB1 == 1 else W1.shape[0])
    if in_feats_len % orig_conv3_out != 0:
        raise RuntimeError(f"in_features ({in_feats_len}) לא מתחלק ב-out_channels של Conv3 לפני גיזום ({orig_conv3_out}).")
    spatial = in_feats_len // orig_conv3_out
    print(f"[Gemm1] transB={transB1}, in_features={in_feats_len}, Conv3_out(orig)={orig_conv3_out}, HxW={spatial}")

    # בונים אינדקסים של in_features לשימור: בלוק רציף של H*W לכל ערוץ שנשמר מ-Conv3
    keep_feat_idx = []
    for c in k3.tolist():
        start = int(c) * spatial
        keep_feat_idx.extend(range(start, start + spatial))
    keep_feat_idx = np.array(keep_feat_idx, dtype=np.int64)

    W1_p = prune_gemm_in_features(W1, keep_feat_idx, transB1)
    W1_init.CopyFrom(numpy_helper.from_array(W1_p, W1_init.name))
    rpt.append({"name": "Gemm1_in", "orig": in_feats_len, "pruned": (W1_p.shape[1] if transB1 == 1 else W1_p.shape[0])})

    # ---- Gemm2 (לוגיטים) – לא נדרש שינוי כל עוד לא גוזרים את out_features של Gemm1 ----
    # ניתן להוסיף בהמשך גיזום גם ל-out_features של Gemm1 ולעדכן בהתאם את BN ואת Gemm2.

    # ---- Shape inference ----
    m = infer_model(m)
    return m, rpt

def main():
    input_model_path = "mnist_emnist_blank_cnn_v1.onnx"
    output_model_path = "mnist_emnist_blank_cnn_v1_pruned.onnx"
    PRUNE_RATIO = 0.4

    print(f"Loading model: {input_model_path}")
    model = onnx.load(input_model_path)

    new_model, report = prune_model(model, prune_ratio=PRUNE_RATIO)

    onnx.save(new_model, output_model_path)
    print("נשמר מודל גזום אל:", output_model_path)

    print("\nPruning report:")
    for r in report:
        print(f"- {r['name']}: {r['orig']} -> {r['pruned']}")

if __name__ == "__main__":
    main()