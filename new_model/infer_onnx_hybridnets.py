import argparse, os, time, math
from typing import List, Tuple
import numpy as np
import cv2
import onnxruntime as ort

# -------- Utilities --------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def parse_size(s: str) -> Tuple[int,int]:
    # format: HxW (e.g., 384x640)
    parts = s.lower().split("x")
    if len(parts) != 2:
        raise ValueError("size must be HxW like 384x640")
    return int(parts[0]), int(parts[1])

def preprocess_bgr(img_bgr: np.ndarray, size_hw: Tuple[int,int], normalize: str):
    H, W = size_hw
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    if normalize == "imagenet":
        img_resized = (img_resized - IMAGENET_MEAN) / IMAGENET_STD
    elif normalize == "none":
        pass
    else:
        raise ValueError("normalize must be 'imagenet' or 'none'")
    # to NCHW
    blob = np.transpose(img_resized, (2, 0, 1))[None, ...].astype(np.float32)
    return blob

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax_channel(x, axis=1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def overlay_mask(img_bgr, mask_prob, color=(0,255,0), alpha=0.4, thresh=0.5):
    mask = (mask_prob >= thresh).astype(np.uint8)
    if mask.ndim == 3:
        # expect HxWxC -> take max
        mask = mask.max(axis=2)
    h, w = img_bgr.shape[:2]
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    overlay = img_bgr.copy()
    overlay[mask > 0] = (overlay[mask > 0] * (1 - alpha) + np.array(color, dtype=np.float32) * alpha).astype(np.uint8)
    return overlay

def nms(boxes, scores, iou_thresh=0.5):
    if len(boxes) == 0:
        return []
    boxes = boxes.astype(np.float32)
    scores = scores.astype(np.float32)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.maximum(x2[i], x2[order[1:]])  # note: using max to be safe; clip later
        yy2 = np.maximum(y2[i], y2[order[1:]])
        # Correct intersection calc:
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = (xx2 - xx1).clip(min=0)
        h = (yy2 - yy1).clip(min=0)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        idx = np.where(iou <= iou_thresh)[0]
        order = order[idx + 1]
    return keep

def try_infer_detection(outputs: List[np.ndarray], orig_shape: Tuple[int,int], score_thresh=0.3, iou_thresh=0.5):
    """
    Heuristic: look for an output shaped like (1, N, >=6) or (N, >=6).
    Interpret as [x1,y1,x2,y2,score,class] if values look reasonable.
    """
    H0, W0 = orig_shape
    det_vis = None
    det_count = 0
    for idx, out in enumerate(outputs):
        arr = out
        # unify shape to (N, C)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 2 and arr.shape[1] >= 6:
            # heuristic sanity: check coordinate ranges
            coords = arr[:, :4]
            scores = arr[:, 4]
            # filter finite
            finite = np.isfinite(coords).all(axis=1) & np.isfinite(scores)
            arr = arr[finite]
            coords = coords[finite]
            scores = scores[finite]
            if arr.shape[0] == 0:
                continue
            # If boxes likely in xyxy already? We'll clip into image range to be safe
            boxes_xyxy = coords.copy()
            # fix cases where coords look like cx,cy,w,h (many negatives earlier indicate raw preds; skip)
            # If too many negatives or widths < 0, skip this output
            wh = boxes_xyxy[:, 2:4] - boxes_xyxy[:, 0:2]
            if (wh < 0).any():
                # doesn't look like xyxy; skip (raw head probably)
                continue
            # scale/clip
            boxes_xyxy[:, 0::2] = boxes_xyxy[:, 0::2].clip(0, W0 - 1)
            boxes_xyxy[:, 1::2] = boxes_xyxy[:, 1::2].clip(0, H0 - 1)
            keep = scores >= score_thresh
            boxes_xyxy = boxes_xyxy[keep]
            scores_kept = scores[keep]
            classes = None
            if arr.shape[1] >= 6:
                # last column as class id (integer if present)
                cls_col = arr[keep, 5]
                # sometimes class is a logit; just cast to int
                classes = cls_col.astype(np.int32)
            # NMS
            keep_idx = nms(boxes_xyxy, scores_kept, iou_thresh=iou_thresh)
            boxes_xyxy = boxes_xyxy[keep_idx]
            scores_kept = scores_kept[keep_idx]
            if classes is not None:
                classes = classes[keep_idx]
            det_count = len(boxes_xyxy)
            det_vis = (boxes_xyxy, scores_kept, classes, idx)
            break
    return det_vis, det_count

# -------- Main Inference --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to hybridnets.onnx")
    ap.add_argument("--image", required=True, help="Path to an input image")
    ap.add_argument("--size", default="384x640", help="Input size HxW (default 384x640)")
    ap.add_argument("--normalize", default="imagenet", choices=["imagenet","none"], help="Input normalization")
    ap.add_argument("--score", type=float, default=0.3, help="Score threshold for detections")
    ap.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    ap.add_argument("--outdir", default="outputs_onnx", help="Output directory")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load model
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(args.model, providers=providers)
    inps = sess.get_inputs()
    outs = sess.get_outputs()
    print("Inputs:")
    for i in inps:
        print(f" - name={i.name} shape={i.shape} type={i.type}")
    print("Outputs:")
    for i, o in enumerate(outs):
        print(f" - [{i}] name={o.name} shape={o.shape} type={o.type}")

    # Read image
    img_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {args.image}")
    H0, W0 = img_bgr.shape[:2]

    H, W = parse_size(args.size)
    blob = preprocess_bgr(img_bgr, (H, W), args.normalize)

    # Run inference
    feed = {inps[0].name: blob}
    t0 = time.time()
    outputs = sess.run(None, feed)
    dt = (time.time() - t0) * 1000
    print(f"Inference time: {dt:.1f} ms ({1000.0/dt if dt>0 else 0:.1f} FPS)")

    # Save raw outputs
    for i, out in enumerate(outputs):
        np.save(os.path.join(args.outdir, f"out_{i}.npy"), out)
        print(f"Saved outputs_onnx/out_{i}.npy shape={out.shape}")

    # Try to find segmentation-like outputs (4D tensors with small C)
    seg_vis_count = 0
    vis_img = img_bgr.copy()
    for i, out in enumerate(outputs):
        arr = out
        if arr.ndim == 4:
            # Expect (N,C,H,W)
            if arr.shape[0] == 1 and arr.shape[1] in (1,2,3) and min(arr.shape[2], arr.shape[3]) >= 16:
                C = arr.shape[1]
                seg = arr[0]
                # Heuristic: sigmoid if single channel, otherwise softmax across channels
                if C == 1:
                    prob = sigmoid(seg[0])
                    mask_overlay = overlay_mask(vis_img, prob, color=(0,255,0), alpha=0.35, thresh=0.5)
                    cv2.imwrite(os.path.join(args.outdir, f"seg_{i}_c0.png"), mask_overlay)
                    seg_vis_count += 1
                else:
                    probs = softmax_channel(seg[None, ...], axis=1)[0]  # back to (C,H,W)
                    colors = [(0,255,0), (0,0,255), (255,0,0)]
                    for c in range(min(C,3)):
                        mask_overlay = overlay_mask(vis_img, probs[c], color=colors[c], alpha=0.35, thresh=0.5)
                        cv2.imwrite(os.path.join(args.outdir, f"seg_{i}_c{c}.png"), mask_overlay)
                        seg_vis_count += 1
                print(f"Rendered segmentation overlays from output {i} (C={C}).")

    # Try to find detection-like outputs
    det_result, det_count = try_infer_detection(outputs, (H0, W0), score_thresh=args.score, iou_thresh=args.iou)
    if det_result is not None:
        boxes_xyxy, scores_kept, classes, src_idx = det_result
        det_img = img_bgr.copy()
        for b, s, c in zip(boxes_xyxy, scores_kept, classes if classes is not None else [None]*len(scores_kept)):
            x1,y1,x2,y2 = b.astype(int)
            cv2.rectangle(det_img, (x1,y1), (x2,y2), (0,255,255), 2)
            label = f"{s:.2f}" if c is None else f"{int(c)}:{s:.2f}"
            cv2.putText(det_img, label, (x1, max(0,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
        out_path = os.path.join(args.outdir, f"dets_from_out_{src_idx}.png")
        cv2.imwrite(out_path, det_img)
        print(f"Drew {det_count} detections from output {src_idx} -> {out_path}")
    else:
        print("No directly-usable detection output found (likely raw head logits). Skipping boxes.")

    # Save a quick visualization base
    cv2.imwrite(os.path.join(args.outdir, "input.jpg"), img_bgr)
    print(f"Done. See folder: {args.outdir}")
    
if __name__ == "__main__":
    main()
