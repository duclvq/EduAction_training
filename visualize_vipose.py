"""
Visualize ViPose keypoint extraction on an EduAction video.
Draws COCO-17 skeleton on each frame and saves as output video.

Usage:
    python visualize_vipose.py                          # random video
    python visualize_vipose.py --video drinking/drinking (1).mp4
    python visualize_vipose.py --class drinking         # random from class
"""

import argparse
import random
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

DATASET_DIR = r"D:\data\EduAction-A-college-student-action-dataset-for-classroom-attention-estimation"
YOLO_MODEL  = r"ViPose\yolo11x-pose.pt"
OUTPUT_DIR  = r"vipose_visualization"
CLASSES     = ["drinking", "lecture", "play_phone", "sleeping",
               "talking", "watch_computer", "writing"]

# COCO-17 skeleton connections
SKELETON = [
    (0,1),(0,2),(1,3),(2,4),       # head
    (5,6),(5,7),(7,9),(6,8),(8,10), # arms
    (5,11),(6,12),(11,12),          # torso
    (11,13),(13,15),(12,14),(14,16) # legs
]
COLORS = {
    "head":   (0,   255, 255),
    "arm":    (0,   255,  0 ),
    "torso":  (255, 128,  0 ),
    "leg":    (0,   128, 255),
}
EDGE_COLORS = [
    COLORS["head"], COLORS["head"], COLORS["head"], COLORS["head"],
    COLORS["arm"],  COLORS["arm"],  COLORS["arm"],  COLORS["arm"], COLORS["arm"],
    COLORS["torso"],COLORS["torso"],COLORS["torso"],
    COLORS["leg"],  COLORS["leg"],  COLORS["leg"],  COLORS["leg"],
]


def draw_pose(frame, keypoints_xy, confidences, conf_thresh=0.3):
    """Draw skeleton on frame. keypoints_xy: (17,2), confidences: (17,)"""
    kps = keypoints_xy.astype(int)

    # Draw bones
    for idx, (i, j) in enumerate(SKELETON):
        if confidences[i] > conf_thresh and confidences[j] > conf_thresh:
            pt1 = tuple(kps[i])
            pt2 = tuple(kps[j])
            color = EDGE_COLORS[idx]
            cv2.line(frame, pt1, pt2, color, 2)

    # Draw joints
    for i, (x, y) in enumerate(kps):
        if confidences[i] > conf_thresh:
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
            cv2.circle(frame, (x, y), 4, (255, 255, 255), 1)

    return frame


def process_video(video_path, yolo, output_path, max_frames=None):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (W, H)
    )

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and frame_count >= max_frames:
            break

        results = yolo(frame, verbose=False, conf=0.15)
        r = results[0]

        if r.keypoints is not None and r.keypoints.xy is not None:
            kps_all  = r.keypoints.xy.cpu().numpy()    # (N, 17, 2)
            conf_all = r.keypoints.conf                 # may be None
            if conf_all is not None:
                conf_all = conf_all.cpu().numpy()       # (N, 17)
            else:
                conf_all = np.ones((len(kps_all), 17)) * 0.9

            for person_kps, person_conf in zip(kps_all, conf_all):
                draw_pose(frame, person_kps, person_conf)

        # Label
        label = f"{video_path.parent.name} | frame {frame_count}"
        cv2.putText(frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 1)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    print(f"Saved: {output_path}  ({frame_count} frames)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=None,
                        help="Relative path inside dataset dir, e.g. 'drinking/drinking (1).mp4'")
    parser.add_argument("--class_name", default=None,
                        help="Pick a random video from this class")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Limit number of frames (default: all)")
    args = parser.parse_args()

    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    # Resolve video path
    if args.video:
        video_path = Path(DATASET_DIR) / args.video
    elif args.class_name:
        cls_dir = Path(DATASET_DIR) / args.class_name
        videos = list(cls_dir.glob("*.mp4"))
        video_path = random.choice(videos)
    else:
        # Pick a random video from a random class
        cls = random.choice(CLASSES)
        videos = list((Path(DATASET_DIR) / cls).glob("*.mp4"))
        video_path = random.choice(videos)

    print(f"Video: {video_path}")

    yolo = YOLO(YOLO_MODEL)

    output_path = Path(OUTPUT_DIR) / f"{video_path.stem}_vipose.mp4"
    process_video(video_path, yolo, output_path, args.max_frames)


if __name__ == "__main__":
    main()
