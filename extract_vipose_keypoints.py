"""
Keypoint Extraction Script for EduAction Dataset using ViPose pipeline.

Two-stage approach:
  1. YOLO11x-pose: person detection + 17-joint COCO pose estimation
  2. ViTPose (optional): keypoint refinement (requires checkpoint file)

Output format matches existing EduAction_pose_data:
  {class}/{video}_keypoints.pkl  -> list of {'frame': int, 'keypoints': np.array(17, 2)}
  {class}/{video}_scores.pkl     -> list of np.array(17,) confidence per frame
  {class}/{video}_metadata.pkl   -> dict with video info

COCO 17 Joints:
  0:Nose  1:L.Eye  2:R.Eye  3:L.Ear  4:R.Ear
  5:L.Shl  6:R.Shl  7:L.Elb  8:R.Elb  9:L.Wrist  10:R.Wrist
  11:L.Hip 12:R.Hip 13:L.Knee 14:R.Knee 15:L.Ankle 16:R.Ankle
"""

import os
import sys
import cv2
import time
import pickle
import argparse
import numpy as np
from pathlib import Path

import torch
from ultralytics import YOLO

# ── ViTPose (optional) ──────────────────────────────────────────────────────
VIPOSE_DIR = Path(__file__).parent / "ViPose"
sys.path.insert(0, str(VIPOSE_DIR))
VITPOSE_AVAILABLE = False
try:
    from model import ViTPose
    import torchvision.transforms as T
    VITPOSE_AVAILABLE = True
except ImportError:
    pass

# ── Defaults ────────────────────────────────────────────────────────────────
DATASET_DIR   = r"D:\data\EduAction-A-college-student-action-dataset-for-classroom-attention-estimation"
OUTPUT_DIR    = r"D:\data\EduAction_vipose_data"
YOLO_MODEL    = str(VIPOSE_DIR / "yolo11x-pose.pt")
VITPOSE_CKPT  = None   # set path if ViTPose checkpoint is available

CLASSES = ["drinking", "lecture", "play_phone", "sleeping", "talking", "watch_computer", "writing"]

FRAME_INTERVAL  = 2      # sample every N frames
YOLO_CONF       = 0.15
IOU_THRESH      = 0.6
CONTAIN_THRESH  = 0.85
VIS_THRESH      = 0.25
VITPOSE_INPUT   = (192, 256)   # (width, height)
PERSON_CLASS_ID = 0

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Helper functions (from run.py) ──────────────────────────────────────────

def filter_boxes(boxes, iou_thresh=0.6, contain_thresh=0.85):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    keep = []
    while boxes:
        cur = boxes.pop(0)
        keep.append(cur)
        remaining = []
        for b in boxes:
            x1 = max(cur[0], b[0]); y1 = max(cur[1], b[1])
            x2 = min(cur[2], b[2]); y2 = min(cur[3], b[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            a1 = (cur[2]-cur[0])*(cur[3]-cur[1])
            a2 = (b[2]-b[0])*(b[3]-b[1])
            union = a1 + a2 - inter
            iou = inter / union if union > 0 else 0
            cont = inter / min(a1, a2) if min(a1, a2) > 0 else 0
            if iou < iou_thresh and cont < contain_thresh:
                remaining.append(b)
        boxes = remaining
    return keep


def box_to_crop(box, img_w, img_h):
    """Expand detected bounding box for full-body coverage."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    ratio = h / max(w, 1)
    new_w = w * 1.25
    mul   = 2.2 if ratio < 1.5 else 3.0
    new_h = max(h * 1.1, new_w * mul)
    new_w = new_h * 0.75
    pad_top = new_h * 0.08
    ny1 = y1 - pad_top
    ny2 = ny1 + new_h
    cx  = x1 + w / 2
    nx1 = cx - new_w / 2
    nx2 = cx + new_w / 2
    return (max(0, int(nx1)), max(0, int(ny1)),
            min(img_w, int(nx2)), min(img_h, int(ny2)))


def refine_with_vitpose(vit_model, crop_img, box_origin):
    """Run ViTPose on a cropped person image; return global (x,y) + scores."""
    transform = T.Compose([
        T.ToPILImage(),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    inp = cv2.resize(crop_img, VITPOSE_INPUT)
    inp = transform(inp).unsqueeze(0)
    if torch.cuda.is_available():
        inp = inp.cuda()
    with torch.no_grad():
        heatmaps = vit_model(inp)
    hm = heatmaps.detach().cpu().numpy()

    ox1, oy1 = box_origin
    cw = crop_img.shape[1]
    ch = crop_img.shape[0]
    kpts, scores = [], []
    for i in range(17):
        _, max_val, _, max_loc = cv2.minMaxLoc(hm[0, i])
        px = max_loc[0] * 4 * (cw / VITPOSE_INPUT[0]) + ox1
        py = max_loc[1] * 4 * (ch / VITPOSE_INPUT[1]) + oy1
        kpts.append([px, py])
        scores.append(float(max_val))
    return np.array(kpts, dtype=np.float32), np.array(scores, dtype=np.float32)


def select_person_center(persons, frame_cx, frame_cy):
    """Pick the person whose torso center is closest to the frame center."""
    best_idx  = 0
    best_dist = float("inf")
    for i, p in enumerate(persons):
        kps  = p["keypoints"]   # (17, 2)
        scs  = p["scores"]      # (17,)
        # Use shoulders + hips for torso center; fall back to bbox center
        anchor_ids = [5, 6, 11, 12]
        valid = [(kps[j], scs[j]) for j in anchor_ids if scs[j] > VIS_THRESH]
        if valid:
            pts  = np.array([v[0] for v in valid])
            cx_p = pts[:, 0].mean()
            cy_p = pts[:, 1].mean()
        else:
            cx_p = p["bbox_cx"]
            cy_p = p["bbox_cy"]
        dist = (cx_p - frame_cx) ** 2 + (cy_p - frame_cy) ** 2
        if dist < best_dist:
            best_dist = dist
            best_idx  = i
    return best_idx


# ── Main extraction logic ────────────────────────────────────────────────────

def process_video(video_path, yolo_model, vit_model, device, frame_interval):
    """
    Extract per-frame keypoints from a single video.

    Returns:
        keypoints_list : list of {'frame': int, 'keypoints': np.ndarray(17,2)}
        scores_list    : list of np.ndarray(17,)
        meta           : dict
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None, None

    fps         = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_cx = W / 2
    frame_cy = H / 2

    keypoints_list = []
    scores_list    = []
    frame_idx      = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            results = yolo_model(frame, verbose=False, conf=YOLO_CONF)
            result  = results[0]

            # ── Collect raw person boxes ─────────────────────────────────
            raw_boxes = []
            for b in result.boxes:
                if int(b.cls[0]) != PERSON_CLASS_ID:
                    continue
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                raw_boxes.append([x1, y1, x2, y2, float(b.conf[0])])

            filtered = filter_boxes(raw_boxes, iou_thresh=IOU_THRESH,
                                    contain_thresh=CONTAIN_THRESH)

            if not filtered:
                # No person detected – store zeros
                keypoints_list.append({
                    "frame": frame_idx,
                    "keypoints": np.zeros((17, 2), dtype=np.float32),
                })
                scores_list.append(np.zeros(17, dtype=np.float32))
                frame_idx += 1
                continue

            # ── Get keypoints for each person ────────────────────────────
            persons = []
            yolo_kps  = result.keypoints  # may be None if YOLO model has no pose head
            yolo_conf = result.keypoints.conf if (yolo_kps is not None and yolo_kps.conf is not None) else None

            for bi, box_data in enumerate(filtered):
                bbox = box_data[:4]

                if vit_model is not None:
                    # Two-stage: YOLO detect → ViTPose refine
                    nx1, ny1, nx2, ny2 = box_to_crop(bbox, W, H)
                    crop = frame[ny1:ny2, nx1:nx2]
                    if crop.size == 0:
                        continue
                    kps, scs = refine_with_vitpose(
                        vit_model, crop, (nx1, ny1)
                    )
                else:
                    # YOLO-only: use built-in pose keypoints
                    # Find which YOLO detection index corresponds to this filtered box
                    # (best effort: match by box area overlap)
                    kps, scs = _match_yolo_kps(result, box_data, W, H)

                persons.append({
                    "keypoints": kps,
                    "scores":    scs,
                    "bbox_cx":   (box_data[0] + box_data[2]) / 2,
                    "bbox_cy":   (box_data[1] + box_data[3]) / 2,
                })

            if not persons:
                keypoints_list.append({
                    "frame": frame_idx,
                    "keypoints": np.zeros((17, 2), dtype=np.float32),
                })
                scores_list.append(np.zeros(17, dtype=np.float32))
                frame_idx += 1
                continue

            best = select_person_center(persons, frame_cx, frame_cy)
            chosen = persons[best]

            keypoints_list.append({
                "frame":     frame_idx,
                "keypoints": chosen["keypoints"].astype(np.float32),
            })
            scores_list.append(chosen["scores"].astype(np.float32))

        frame_idx += 1

    cap.release()

    meta = {
        "video_path":    str(video_path),
        "video_name":    video_path.stem,
        "class":         video_path.parent.name,
        "fps":           fps,
        "frame_width":   W,
        "frame_height":  H,
        "total_frames":  total_frames,
        "sampled_frames": len(keypoints_list),
        "frame_interval": frame_interval,
        "n_joints":      17,
        "joint_format":  "COCO17",
        "selection_strategy": "center",
        "extractor":     "ViTPose" if vit_model is not None else "YOLO11x-pose",
    }
    return keypoints_list, scores_list, meta


def _match_yolo_kps(result, filtered_box, W, H):
    """
    Match a filtered box back to a YOLO detection and return its keypoints.
    Falls back to zeros if no match or YOLO pose output unavailable.
    """
    if result.keypoints is None or result.keypoints.xy is None:
        return np.zeros((17, 2), dtype=np.float32), np.zeros(17, dtype=np.float32)

    all_xy   = result.keypoints.xy.cpu().numpy()    # (N, 17, 2)
    all_conf = result.keypoints.conf                 # may be None
    if all_conf is not None:
        all_conf = all_conf.cpu().numpy()            # (N, 17)
    else:
        all_conf = np.ones((len(all_xy), 17), dtype=np.float32) * 0.5

    all_boxes = result.boxes.xyxy.cpu().numpy()  # (N, 4)

    fx1, fy1, fx2, fy2 = filtered_box[:4]
    best_idx = 0
    best_iou = -1.0
    for i, bx in enumerate(all_boxes):
        ix1 = max(fx1, bx[0]); iy1 = max(fy1, bx[1])
        ix2 = min(fx2, bx[2]); iy2 = min(fy2, bx[3])
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        a1 = (fx2-fx1)*(fy2-fy1)
        a2 = (bx[2]-bx[0])*(bx[3]-bx[1])
        iou = inter / (a1+a2-inter+1e-6)
        if iou > best_iou:
            best_iou = iou
            best_idx = i

    return (all_xy[best_idx].astype(np.float32),
            all_conf[best_idx].astype(np.float32))


# ── Entry point ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract ViPose keypoints from EduAction dataset videos"
    )
    parser.add_argument("--dataset_dir", default=DATASET_DIR,
                        help="Root folder of EduAction dataset")
    parser.add_argument("--output_dir",  default=OUTPUT_DIR,
                        help="Where to save extracted keypoints")
    parser.add_argument("--yolo_model",  default=YOLO_MODEL,
                        help="Path to YOLO11x-pose.pt")
    parser.add_argument("--vitpose_ckpt", default=VITPOSE_CKPT,
                        help="Path to ViTPose checkpoint (.pth). "
                             "If not set, uses YOLO-only mode.")
    parser.add_argument("--frame_interval", type=int, default=FRAME_INTERVAL,
                        help="Sample every N frames (default: 2)")
    parser.add_argument("--classes", nargs="+", default=CLASSES,
                        help="Classes to process")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip videos that already have output files")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_vitpose(ckpt_path, device):
    """Load ViTPose model from checkpoint."""
    if not VITPOSE_AVAILABLE:
        print("[WARN] ViTPose module not importable. Using YOLO-only mode.")
        return None
    if ckpt_path is None or not Path(ckpt_path).exists():
        print(f"[WARN] ViTPose checkpoint not found: {ckpt_path}. Using YOLO-only mode.")
        return None

    print(f"[INFO] Loading ViTPose from {ckpt_path}")
    model = ViTPose(model_size="base", num_joints=17, img_size=(256, 192))
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    model.to(device)
    return model


def main():
    args   = parse_args()
    device = torch.device(args.device)

    print(f"[INFO] Dataset : {args.dataset_dir}")
    print(f"[INFO] Output  : {args.output_dir}")
    print(f"[INFO] Device  : {device}")
    print(f"[INFO] Frame interval: every {args.frame_interval} frames")

    # Load models
    print(f"[INFO] Loading YOLO from {args.yolo_model}")
    yolo_model = YOLO(args.yolo_model)

    vit_model  = load_vitpose(args.vitpose_ckpt, device)
    mode_str   = "YOLO11x-pose + ViTPose" if vit_model else "YOLO11x-pose only"
    print(f"[INFO] Extraction mode: {mode_str}")

    total_processed = 0
    total_skipped   = 0
    total_errors    = 0
    t0 = time.time()

    for cls_name in args.classes:
        cls_dir    = Path(args.dataset_dir) / cls_name
        out_cls    = Path(args.output_dir)  / cls_name
        out_cls.mkdir(parents=True, exist_ok=True)

        videos = sorted(cls_dir.glob("*.mp4"))
        if not videos:
            print(f"[WARN] No MP4 files in {cls_dir}")
            continue

        print(f"\n[CLASS] {cls_name} — {len(videos)} videos")

        for vi, video_path in enumerate(videos):
            stem       = video_path.stem
            out_kp     = out_cls / f"{stem}_keypoints.pkl"
            out_sc     = out_cls / f"{stem}_scores.pkl"
            out_meta   = out_cls / f"{stem}_metadata.pkl"

            if args.skip_existing and out_kp.exists():
                print(f"  [{vi+1}/{len(videos)}] SKIP {stem}")
                total_skipped += 1
                continue

            t1 = time.time()
            try:
                kp_list, sc_list, meta = process_video(
                    video_path, yolo_model, vit_model, device, args.frame_interval
                )
            except Exception as e:
                print(f"  [{vi+1}/{len(videos)}] ERROR {stem}: {e}")
                total_errors += 1
                continue

            if kp_list is None:
                print(f"  [{vi+1}/{len(videos)}] FAIL  {stem} (cannot open video)")
                total_errors += 1
                continue

            with open(out_kp,   "wb") as f: pickle.dump(kp_list, f)
            with open(out_sc,   "wb") as f: pickle.dump(sc_list, f)
            with open(out_meta, "wb") as f: pickle.dump(meta,    f)

            elapsed = time.time() - t1
            print(f"  [{vi+1}/{len(videos)}] OK    {stem}"
                  f" | frames={meta['sampled_frames']}"
                  f" | {elapsed:.1f}s")
            total_processed += 1

    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Done. Processed: {total_processed}, Skipped: {total_skipped}, Errors: {total_errors}")
    print(f"Total time: {total_time/60:.1f} min")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
