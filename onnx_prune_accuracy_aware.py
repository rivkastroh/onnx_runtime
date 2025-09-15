import os
import glob
import math
import copy
import argparse
from collections import defaultdict, deque

import onnx
import numpy as np
from onnx import numpy_helper, shape_inference

# ננסה להשתמש ב-onnxruntime עבור כיול אקטיבציות; אם לא קיים - נמשיך בלי כיול
try:
    import onnxruntime as ort  # type: ignore
    ORT_AVAILABLE = True
except Exception:
    ORT_AVAILABLE = False


# ------------------------- עזר בסיסי -------------------------

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

def build_consumers_map(model):
    consumers = defaultdict(list)  # tensor_name -> [nodes that consume it]
    for node in model.graph.node:
        for inp in node.input:
            if inp:
                consumers[inp].append(node)
    return consumers

# אופרטורים שמעבירים את מספר הערוצים C ללא שינוי
CHANNEL_PRESERVING = {
    "Relu", "LeakyRelu", "PRelu", "Sigmoid", "Tanh",
    "BatchNormalization", "Identity", "Cast", "Clip",
    "MaxPool", "AveragePool", "GlobalMaxPool", "GlobalAveragePool",
}

def ensure_graph_outputs(model, tensor_names):
    # דואג שכל שמות ה-tensor ברשימה יהיו outputs של הגרף כדי שאפשר יהיה לשלוף אותם עם ORT
    existing = {o.name for o in model.graph.output}
    for name in tensor_names:
        if name in existing:
            continue
        # ננסה למצוא מידע ב-value_info כדי לשמר טיפוס/צורה
        vi = None
        for v in list(model.graph.value_info) + list(model.graph.output) + list(model.graph.input):
            if v.name == name:
                vi = v
                break
        if vi is not None:
            model.graph.output.append(copy.deepcopy(vi))
        else:
            # ניצור ValueInfo בלי צורה - ORT מסתדר לאחר infer_shapes
            from onnx import helper, TensorProto
            vi = helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
            model.graph.output.append(vi)
    return model

def get_model_input_spec(model):
    # מחזיר שם קלט וצורתו (N,C,H,W אם יש). שימושי לכיול
    if not model.graph.input:
        raise RuntimeError("אין קלטים לגרף.")
    inp = model.graph.input[0]
    name = inp.name
    # ננסה לשלוף צורה
    shape = None
    try:
        dims = inp.type.tensor_type.shape.dim
        shape = []
        for d in dims:
            if d.dim_value > 0:
                shape.append(int(d.dim_value))
            else:
                # dim_param או דינמי - נסמן כ-None
                shape.append(None)
    except Exception:
        pass
    return name, shape

def load_images_for_calibration(calib_dir, target_ch, target_h, target_w, max_samples=64):
    # טוען תמונות מתיקייה, ממיר ל-float32 [0,1], משכפל ערוץ אם צריך, ומחזיר np.ndarray בצורה [N,C,H,W]
    from PIL import Image  # lazy
    paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        paths.extend(glob.glob(os.path.join(calib_dir, ext)))
    paths = sorted(paths)[:max_samples]
    if not paths:
        raise RuntimeError(f"לא נמצאו תמונות בנתיב כיול: {calib_dir}")

    imgs = []
    for p in paths:
        img = Image.open(p).convert("L" if target_ch == 1 else "RGB")
        img = img.resize((target_w, target_h))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        if arr.ndim == 2:  # H,W
            arr = arr[None, :, :]  # 1,H,W
        else:  # H,W,C
            arr = arr.transpose(2, 0, 1)  # C,H,W
            if target_ch == 1:
                # המרת RGB לאפור ממוצע פשוט
                arr = arr.mean(axis=0, keepdims=True)
            elif arr.shape[0] > target_ch:
                arr = arr[:target_ch, :, :]
            elif arr.shape[0] < target_ch:
                # שכפול ערוצים אם צריך
                arr = np.repeat(arr, target_ch // arr.shape[0], axis=0)
                if arr.shape[0] != target_ch:
                    arr = arr[:target_ch, :, :]
        imgs.append(arr)
    batch = np.stack(imgs, axis=0)  # N,C,H,W
    return batch

# ------------------------- גיזום -------------------------

def l1_per_filter(weights):  # [out, in, kH, kW]
    return np.sum(np.abs(weights), axis=(1, 2, 3))

def normalized(x, eps=1e-9):
    x = np.asarray(x, dtype=np.float32)
    m = np.mean(x)
    s = np.std(x) + eps
    return (x - m) / s

def importance_from_weights_and_acts(w_np, acts_mean_abs=None, alpha=0.7):
    # חשיבות משוקללת: alpha*אקטיבציות + (1-alpha)*נורמת משקל
    imp_w = l1_per_filter(w_np)
    imp_w = normalized(imp_w)
    if acts_mean_abs is None:
        return imp_w  # ללא כיול
    ama = np.asarray(acts_mean_abs, dtype=np.float32)
    ama = normalized(ama)
    return alpha * ama + (1.0 - alpha) * imp_w

def choose_keep_indices(importance, out_channels, prune_ratio, min_keep=1, multiple_of=1):
    keep_target = int(round((1.0 - prune_ratio) * out_channels))
    keep_target = max(keep_target, min_keep)
    if multiple_of > 1:
        # עיגול כלפי מטה למכפלה קרובה
        keep_target = max(multiple_of, (keep_target // multiple_of) * multiple_of)
        keep_target = min(keep_target, out_channels)  # לא לחרוג
    keep_target = max(1, min(keep_target, out_channels))
    order = np.argsort(-importance)  # גדול קודם
    keep = order[:keep_target]
    keep.sort()
    return keep

def prune_conv_layer(model, conv_node, keep_out_idx):
    w_init = get_initializer(model, conv_node.input[1])
    b_init = get_initializer(model, conv_node.input[2]) if len(conv_node.input) > 2 else None
    w = numpy_helper.to_array(w_init)
    w_new = w[keep_out_idx, :, :, :]
    w_init.CopyFrom(numpy_helper.from_array(w_new, w_init.name))
    if b_init is not None:
        b = numpy_helper.to_array(b_init)
        b_new = b[keep_out_idx]
        b_init.CopyFrom(numpy_helper.from_array(b_new, b_init.name))

def update_downstream_convs_input_channels(model, start_tensor, keep_out_idx, consumers_map):
    q = deque([start_tensor])
    visited = set([start_tensor])
    while q:
        tname = q.popleft()
        for node in consumers_map.get(tname, []):
            if node.op_type == "Conv":
                w_init = get_initializer(model, node.input[1])
                if w_init is None:
                    continue
                w = numpy_helper.to_array(w_init)
                w_new = w[:, keep_out_idx, :, :]
                w_init.CopyFrom(numpy_helper.from_array(w_new, w_init.name))
                print(f"עודכן in_channels של Conv {node.name or node.output[0]} ל-{w_new.shape[1]}")
                # לא ממשיכים אחרי Conv
            elif node.op_type in CHANNEL_PRESERVING:
                for out_t in node.output:
                    if out_t and out_t not in visited:
                        visited.add(out_t)
                        q.append(out_t)
            else:
                # מפעילים אחרים (Flatten/Reshape/Concat וכו') – עצירה
                pass

def prune_gemm1_in_features(model, gemm_node, conv3_keep_idx, conv3_out_channels_before):
    # חתך קלטי Gemm לפי ערוצי Conv3 שנשמרו (בלוקים של H*W לכל ערוץ)
    transB = get_attr_i(gemm_node, "transB", 1)  # בדרך כלל 1
    W_init = get_initializer(model, gemm_node.input[1])
    if W_init is None:
        raise RuntimeError("לא נמצאו משקולות ל-Gemm1.")
    W = numpy_helper.to_array(W_init)

    in_feats_len = int(W.shape[1] if transB == 1 else W.shape[0])
    if in_feats_len % conv3_out_channels_before != 0:
        raise RuntimeError(f"in_features ({in_feats_len}) לא מתחלק ב-out_channels של Conv3 לפני גיזום ({conv3_out_channels_before}).")
    spatial = in_feats_len // conv3_out_channels_before

    keep_feat_idx = []
    for c in conv3_keep_idx.tolist():
        start = int(c) * spatial
        keep_feat_idx.extend(range(start, start + spatial))
    keep_feat_idx = np.array(keep_feat_idx, dtype=np.int64)

    if transB == 1:
        W_new = W[:, keep_feat_idx]
    else:
        W_new = W[keep_feat_idx, :]
    W_init.CopyFrom(numpy_helper.from_array(W_new, W_init.name))
    print(f"עודכן in_features של Gemm1 ל-{W_new.shape[1] if transB==1 else W_new.shape[0]} (HxW={spatial})")

# ------------------------- כיול אקטיבציות (אופציונלי) -------------------------

def collect_activation_means(model, output_names, calib_dir, max_samples=64):
    """
    מחזיר dict: tensor_name -> mean_abs per-channel (וקטור בגודל C)
    """
    if not ORT_AVAILABLE:
        print("אזהרה: onnxruntime לא זמין – מדלגים על כיול אקטיבציות.")
        return {}

    # נבנה עותק עם outputs נוספים
    inst_model = copy.deepcopy(model)
    inst_model = infer_model(inst_model)
    inst_model = ensure_graph_outputs(inst_model, output_names)
    inst_model = infer_model(inst_model)

    # כנה קלט
    inp_name, inp_shape = get_model_input_spec(inst_model)
    # נזהה C,H,W
    if inp_shape is None or len(inp_shape) < 4:
        raise RuntimeError("לא ניתן לזהות צורת קלט (צפוי NCHW).")
    N, C, H, W = inp_shape[:4]
    if C is None or H is None or W is None:
        # ננסה להיעזר ב-value_info אחרי infer_shapes
        # אם עדיין אין, נדרוש מהמשתמש לספק ידנית דרך פרמטרים (פשטות)
        raise RuntimeError("צורת קלט עם ממדים דינמיים – ספקי ידנית או שמרו מודל עם צורה סטטית.")

    batch = load_images_for_calibration(calib_dir, target_ch=C, target_h=H, target_w=W, max_samples=max_samples)

    sess = ort.InferenceSession(onnx._serialize(inst_model), providers=["CPUExecutionProvider"])  # type: ignore
    results_sum = {name: None for name in output_names}
    counts = 0

    # נריץ במנות קטנות כדי לא לחרוג מזיכרון
    step = max(1, min(16, batch.shape[0]))
    for i in range(0, batch.shape[0], step):
        x = batch[i:i+step]
        outs = sess.run(output_names, {inp_name: x})
        for name, arr in zip(output_names, outs):
            # נניח תפוקה בצורה [N,C,H,W]
            if arr.ndim < 2:
                continue
            # ממוצע ערכי מוחלט פר-ערוץ
            if arr.ndim == 4:
                ama = np.mean(np.abs(arr), axis=(0, 2, 3))  # C
            elif arr.ndim == 2:
                ama = np.mean(np.abs(arr), axis=0)  # C
            else:
                # צורות אחרות – נממש רק מקרה הנפוץ
                ama = np.mean(np.abs(arr), axis=0)
            if results_sum[name] is None:
                results_sum[name] = np.array(ama, dtype=np.float64)
            else:
                results_sum[name] += ama
        counts += 1

    mean_abs = {name: (results_sum[name] / counts) if results_sum[name] is not None else None
                for name in output_names}
    return mean_abs


# ------------------------- גיזום מודע-דיוק -------------------------

def prune_model_accuracy_aware(model,
                               prune_ratios=(0.10, 0.20, 0.30),
                               min_channels=(24, 48, 96),
                               alpha=0.7,
                               calib_dir=None,
                               multiple_of=8,
                               verbose=True):
    """
    prune_ratios: יחס גיזום לכל Conv (Conv1, Conv2, Conv3)
    min_channels: מינימום ערוצים להשאיר לכל Conv (במקור: 32,64,128)
    alpha: משקל לאקטיבציות לעומת משקלים (0..1)
    calib_dir: אם ניתן – יבצע כיול אקטיבציות וישפר דיוק
    multiple_of: עיגול מספר הערוצים שנשמרו למכפלה (לשיפור ביצועים ויציבות)
    """
    m = copy.deepcopy(model)
    m = infer_model(m)
    consumers_map = build_consumers_map(m)

    # שמות מהגרף שלך:
    CONV1_OUT = "/features/features.0/Conv_output_0"
    RELU1_OUT = "/features/features.2/Relu_output_0"
    CONV2_OUT = "/features/features.3/Conv_output_0"
    RELU2_OUT = "/features/features.5/Relu_output_0"
    CONV3_OUT = "/features/features.8/Conv_output_0"
    RELU3_OUT = "/features/features.10/Relu_output_0"

    # Gemm1 (אחרי Flatten)
    GEMM1_W = "classifier.1.weight"
    GEMM1_B = "classifier.1.bias"

    # 1) איסוף אקטיבציות (אופציונלי אך מומלץ)
    acts_means = {}
    if calib_dir:
        try:
            acts_means = collect_activation_means(m, [RELU1_OUT, RELU2_OUT, RELU3_OUT], calib_dir, max_samples=64)
        except Exception as e:
            print(f"אזהרה: כיול אקטיבציות נכשל ({e}) – נמשיך עם גיזום לפי משקלים בלבד.")
            acts_means = {}

    # 2) Conv1
    conv1 = get_node_by_output(m, CONV1_OUT)
    if conv1 is None:
        raise RuntimeError("Conv1 לא נמצא לפי הפלט.")
    w1_init = get_initializer(m, conv1.input[1])
    b1_init = get_initializer(m, conv1.input[2]) if len(conv1.input) > 2 else None
    w1 = numpy_helper.to_array(w1_init)
    out1 = w1.shape[0]
    imp1 = importance_from_weights_and_acts(w1, acts_means.get(RELU1_OUT), alpha=alpha)
    k1 = choose_keep_indices(imp1, out1, prune_ratios[0], min_keep=min_channels[0], multiple_of=multiple_of)
    prune_conv_layer(m, conv1, k1)
    if verbose:
        print(f"Conv1: {out1} -> {len(k1)} (ratio={prune_ratios[0]:.2f})")
    # עדכון צרכנים בהמשך
    update_downstream_convs_input_channels(m, conv1.output[0], k1, consumers_map)

    # 3) Conv2
    conv2 = get_node_by_output(m, CONV2_OUT)
    if conv2 is None:
        raise RuntimeError("Conv2 לא נמצא לפי הפלט.")
    w2_init = get_initializer(m, conv2.input[1])
    w2 = numpy_helper.to_array(w2_init)
    out2 = w2.shape[0]
    imp2 = importance_from_weights_and_acts(w2, acts_means.get(RELU2_OUT), alpha=alpha)
    k2 = choose_keep_indices(imp2, out2, prune_ratios[1], min_keep=min_channels[1], multiple_of=multiple_of)
    prune_conv_layer(m, conv2, k2)
    if verbose:
        print(f"Conv2: {out2} -> {len(k2)} (ratio={prune_ratios[1]:.2f})")
    update_downstream_convs_input_channels(m, conv2.output[0], k2, consumers_map)

    # 4) Conv3
    conv3 = get_node_by_output(m, CONV3_OUT)
    if conv3 is None:
        raise RuntimeError("Conv3 לא נמצא לפי הפלט.")
    w3_init = get_initializer(m, conv3.input[1])
    w3_before = numpy_helper.to_array(w3_init)  # לפני גיזום (צריך בשביל Gemm1)
    out3 = w3_before.shape[0]
    imp3 = importance_from_weights_and_acts(w3_before, acts_means.get(RELU3_OUT), alpha=alpha)
    k3 = choose_keep_indices(imp3, out3, prune_ratios[2], min_keep=min_channels[2], multiple_of=multiple_of)
    prune_conv_layer(m, conv3, k3)
    if verbose:
        print(f"Conv3: {out3} -> {len(k3)} (ratio={prune_ratios[2]:.2f})")

    # 5) לעדכן Gemm1 in_features אחרי Flatten
    # מצא את Gemm1 לפי ה-weight
    gemm_nodes = get_nodes_by_type(m, "Gemm")
    gemm1 = None
    for n in gemm_nodes:
        wi = get_initializer(m, n.input[1]) if len(n.input) > 1 else None
        if wi is not None and wi.name == GEMM1_W:
            gemm1 = n
            break
    if gemm1 is None:
        # fallback: הראשון
        gemm1 = gemm_nodes[0] if gemm_nodes else None
    if gemm1 is None:
        raise RuntimeError("לא נמצא Gemm1 במודל.")

    prune_gemm1_in_features(m, gemm1, conv3_keep_idx=k3, conv3_out_channels_before=out3)

    # 6) Shape inference
    m = infer_model(m)
    return m


# ------------------------- CLI -------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Prune ONNX CNN accuracy-aware (ערוץ-חכם)")
    ap.add_argument("--input", required=True, help="קובץ ONNX קלט")
    ap.add_argument("--output", required=True, help="קובץ ONNX פלט אחרי גיזום")
    ap.add_argument("--prune-ratios", default="0.10,0.20,0.30",
                    help="יחסי גיזום ל-Conv1,Conv2,Conv3 (למשל: 0.1,0.2,0.3)")
    ap.add_argument("--min-channels", default="24,48,96",
                    help="מינימום ערוצים להשאיר ל-Conv1,Conv2,Conv3 (למשל: 24,48,96)")
    ap.add_argument("--alpha", type=float, default=0.7,
                    help="משקל לאקטיבציות מול משקלים (0..1). גבוה יותר = מסתמך יותר על כיול אקטיבציות")
    ap.add_argument("--multiple-of", type=int, default=8,
                    help="לעגל את מספר הערוצים שנשמרו למכפלה זו (משפר יציבות/ביצועים)")
    ap.add_argument("--calib-dir", default="",
                    help="תיקיית כיול עם תמונות (אם לא מסופק/אין ORT – הסקריפט יעבוד ללא כיול)")
    return ap.parse_args()

def main():
    args = parse_args()

    prune_ratios = tuple(float(x.strip()) for x in args.prune_ratios.split(","))
    min_channels = tuple(int(x.strip()) for x in args.min_channels.split(","))
    if len(prune_ratios) != 3 or len(min_channels) != 3:
        raise ValueError("יש לספק בדיוק 3 ערכים ל--prune-ratios ול--min-channels.")

    print(f"טעינת מודל: {args.input}")
    model = onnx.load(args.input)

    calib_dir = args.calib_dir if args.calib_dir and os.path.isdir(args.calib_dir) else None
    if args.calib_dir and calib_dir is None:
        print(f"אזהרה: תיקיית כיול לא קיימת: {args.calib_dir} – נמשיך ללא כיול.")

    pruned = prune_model_accuracy_aware(
        model,
        prune_ratios=prune_ratios,
        min_channels=min_channels,
        alpha=args.alpha,
        calib_dir=calib_dir,
        multiple_of=args.multiple_of,
        verbose=True,
    )

    onnx.save(pruned, args.output)
    print(f"נשמר מודל גזום אל: {args.output}")

if __name__ == "__main__":
    main()