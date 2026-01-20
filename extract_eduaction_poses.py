"""
Extract pose keypoints from EduAction dataset videos.
This script processes videos in the EduAction dataset, extracts pose keypoints,
and filters to keep only the central or largest person in frame.
"""

import os
import cv2
import numpy as np
from rtmlib import Wholebody
import time
import pickle
from tqdm import tqdm
from pathlib import Path


def calculate_bbox_area(keypoints):
    """
    Calculate bounding box area from keypoints.

    Args:
        keypoints: numpy array of shape (N, 2) containing x, y coordinates

    Returns:
        float: area of bounding box
    """
    if len(keypoints) == 0:
        return 0

    min_x, min_y = np.min(keypoints, axis=0)
    max_x, max_y = np.max(keypoints, axis=0)

    width = max_x - min_x
    height = max_y - min_y

    return width * height


def calculate_bbox_center(keypoints):
    """
    Calculate center of bounding box from keypoints.

    Args:
        keypoints: numpy array of shape (N, 2) containing x, y coordinates

    Returns:
        tuple: (center_x, center_y)
    """
    if len(keypoints) == 0:
        return (0, 0)

    min_x, min_y = np.min(keypoints, axis=0)
    max_x, max_y = np.max(keypoints, axis=0)

    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    return (center_x, center_y)


def calculate_keypoint_center(keypoints):
    """
    Calculate the average center of valid keypoints (not bbox center).
    This gives better representation of actual person position.

    Args:
        keypoints: numpy array of shape (N, 2) containing x, y coordinates

    Returns:
        tuple: (center_x, center_y) - average position of all valid keypoints
    """
    # Filter out zero/invalid keypoints
    valid_kpts = keypoints[(keypoints[:, 0] != 0) & (keypoints[:, 1] != 0)]

    if len(valid_kpts) == 0:
        return (0, 0)

    # Return mean of all valid keypoints
    center_x = np.mean(valid_kpts[:, 0])
    center_y = np.mean(valid_kpts[:, 1])

    return (center_x, center_y)


def select_main_person(keypoints_list, scores_list, frame_shape, strategy='center'):
    """
    Select the main person from multiple detected persons.
    Default strategy is 'center' - prioritizes person closest to frame center.

    Args:
        keypoints_list: list of keypoints arrays for each detected person
        scores_list: list of confidence scores for each detected person
        frame_shape: tuple of (height, width) of the video frame
        strategy: 'center' (default, best for classroom), 'largest', or 'combined'

    Returns:
        tuple: (selected_keypoints, selected_score) or (None, None) if no person detected
    """
    if len(keypoints_list) == 0:
        return None, None

    if len(keypoints_list) == 1:
        return keypoints_list[0], scores_list[0]

    frame_height, frame_width = frame_shape[:2]
    frame_center = (frame_width / 2, frame_height / 2)

    if strategy == 'center':
        # Select person whose average keypoint position is closest to frame center
        # This uses actual pose location, not bbox
        distances = []
        for kpts in keypoints_list:
            kpt_center = calculate_keypoint_center(kpts)

            # Skip if no valid keypoints
            if kpt_center == (0, 0):
                distances.append(float('inf'))
                continue

            dist = np.sqrt((kpt_center[0] - frame_center[0])**2 +
                          (kpt_center[1] - frame_center[1])**2)
            distances.append(dist)

        min_idx = np.argmin(distances)
        return keypoints_list[min_idx], scores_list[min_idx]

    elif strategy == 'largest':
        # Select person with largest bounding box area
        areas = [calculate_bbox_area(kpts) for kpts in keypoints_list]
        max_idx = np.argmax(areas)
        return keypoints_list[max_idx], scores_list[max_idx]

    elif strategy == 'combined':
        # Combined strategy: prioritize center position (70%), area is secondary (30%)
        scores = []
        for kpts in keypoints_list:
            area = calculate_bbox_area(kpts)
            kpt_center = calculate_keypoint_center(kpts)

            # Skip if no valid keypoints
            if kpt_center == (0, 0):
                scores.append(-1)
                continue

            dist = np.sqrt((kpt_center[0] - frame_center[0])**2 +
                          (kpt_center[1] - frame_center[1])**2)

            # Normalize distance (0-1, where 0 is at center)
            max_dist = np.sqrt(frame_width**2 + frame_height**2) / 2
            norm_dist = dist / max_dist

            # Normalize area (0-1)
            max_area = frame_width * frame_height
            norm_area = area / max_area

            # Score: prioritize center (70%), area (30%)
            score = (1 - norm_dist) * 0.7 + norm_area * 0.3
            scores.append(score)

        max_idx = np.argmax(scores)
        return keypoints_list[max_idx], scores_list[max_idx]

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def process_eduaction_dataset(
    dataset_dir,
    output_dir,
    selection_strategy='center',
    backend='onnxruntime',
    device='cuda',
    openpose_skeleton=False
):
    """
    Process all videos in EduAction dataset and extract pose keypoints.

    Args:
        dataset_dir: path to EduAction dataset root directory
        output_dir: path to save extracted pose data
        selection_strategy: 'largest', 'center', or 'combined'
        backend: 'opencv', 'onnxruntime', or 'openvino'
        device: 'cpu', 'cuda', or 'mps'
        openpose_skeleton: True for openpose-style, False for mmpose-style
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize pose detector
    print(f"Initializing Wholebody detector (device: {device}, backend: {backend})...")
    wholebody = Wholebody(
        to_openpose=openpose_skeleton,
        mode='balanced',
        backend=backend,
        device=device
    )

    # Get all class folders
    class_folders = [f for f in os.listdir(dataset_dir)
                    if os.path.isdir(os.path.join(dataset_dir, f)) and f != '__pycache__']
    class_folders.sort()

    print(f"Found {len(class_folders)} classes: {class_folders}")
    print(f"Selection strategy: {selection_strategy}\n")

    # Process each class
    for cls in class_folders:
        cls_path = os.path.join(dataset_dir, cls)

        # Get all video files in this class folder
        video_files = [f for f in os.listdir(cls_path) if f.endswith(('.mp4', '.avi', '.mov'))]

        if len(video_files) == 0:
            print(f"No video files found in {cls_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing class: {cls} ({len(video_files)} videos)")
        print(f"{'='*60}")

        # Create output directory for this class
        cls_output_dir = os.path.join(output_dir, cls)
        os.makedirs(cls_output_dir, exist_ok=True)

        # Process each video in this class
        for video_file in tqdm(video_files, desc=f"Class {cls}"):
            video_path = os.path.join(cls_path, video_file)
            video_name = Path(video_file).stem  # filename without extension

            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"\nError: Cannot open video {video_path}")
                continue

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Storage for keypoints and scores
            all_keypoints = []
            all_scores = []
            frame_num = 0

            # Process each frame
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Detect poses
                keypoints, scores = wholebody(frame)

                # Select main person
                selected_keypoints, selected_score = select_main_person(
                    keypoints, scores, frame.shape, strategy=selection_strategy
                )

                # Store results
                all_keypoints.append({
                    'frame': frame_num,
                    'keypoints': selected_keypoints  # Can be None if no person detected
                })
                all_scores.append(selected_score)  # Can be None if no person detected

                frame_num += 1

            cap.release()

            # Save keypoints and scores
            output_keypoints_path = os.path.join(cls_output_dir, f"{video_name}_keypoints.pkl")
            output_scores_path = os.path.join(cls_output_dir, f"{video_name}_scores.pkl")

            with open(output_keypoints_path, 'wb') as f:
                pickle.dump(all_keypoints, f)

            with open(output_scores_path, 'wb') as f:
                pickle.dump(all_scores, f)

            # Save metadata
            metadata = {
                'video_path': video_path,
                'video_name': video_name,
                'class': cls,
                'fps': fps,
                'frame_width': frame_width,
                'frame_height': frame_height,
                'total_frames': frame_num,
                'selection_strategy': selection_strategy
            }

            metadata_path = os.path.join(cls_output_dir, f"{video_name}_metadata.pkl")
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)

    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"Output saved to: {output_dir}")
    print(f"{'='*60}")


def main():
    # Configuration
    dataset_dir = r"D:\data\EduAction-A-college-student-action-dataset-for-classroom-attention-estimation"
    output_dir = r"D:\data\EduAction_pose_data"

    # Processing parameters
    # IMPORTANT: Changed default to 'center' to prioritize person at center of frame
    # This avoids selecting people at edges/background
    selection_strategy = 'center'  # 'center' (recommended), 'largest', or 'combined'
    backend = 'onnxruntime'  # 'opencv', 'onnxruntime', 'openvino'
    device = 'cuda'  # 'cpu', 'cuda', 'mps'
    openpose_skeleton = False  # True for openpose-style, False for mmpose-style

    # Run processing
    process_eduaction_dataset(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        selection_strategy=selection_strategy,
        backend=backend,
        device=device,
        openpose_skeleton=openpose_skeleton
    )


if __name__ == "__main__":
    main()
