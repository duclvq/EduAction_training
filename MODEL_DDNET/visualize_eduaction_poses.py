"""
Visualize pose keypoints on EduAction videos.
This script overlays extracted keypoints on the original videos for verification.
Supports different keypoint configurations to match training configs.
"""

import os
import pickle
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random


class KeypointConfig:
    """Configuration for which keypoints to visualize"""

    def __init__(self, use_legs=True, use_face=True, use_hands=True):
        self.use_legs = use_legs
        self.use_face = use_face
        self.use_hands = use_hands

        # Build keypoint indices
        self.keypoint_indices = []

        # Body keypoints
        if use_legs:
            self.keypoint_indices.extend(range(0, 23))  # Full body + feet
        else:
            self.keypoint_indices.extend(range(0, 13))  # Upper body only

        # Face keypoints
        if use_face:
            self.keypoint_indices.extend(range(23, 91))

        # Hand keypoints
        if use_hands:
            self.keypoint_indices.extend(range(91, 133))

        self.name = self._generate_name()

    def _generate_name(self):
        """Generate config name"""
        parts = []
        if self.use_legs:
            parts.append("Legs")
        if self.use_face:
            parts.append("Face")
        if self.use_hands:
            parts.append("Hands")
        return "+".join(parts) if parts else "Body Only"

    def filter_keypoints(self, keypoints):
        """Filter keypoints based on config"""
        if len(self.keypoint_indices) == 133:
            return keypoints  # Use all

        # Select only configured keypoints
        if keypoints.shape[0] >= max(self.keypoint_indices) + 1:
            return keypoints[self.keypoint_indices, :]
        else:
            return keypoints


def draw_skeleton_connections(frame, keypoints, kp_config=None, color=(0, 255, 0), thickness=2):
    """
    Draw skeleton connections between keypoints.

    Args:
        frame: video frame to draw on
        keypoints: array of shape (N, 2) containing x, y coordinates
        kp_config: KeypointConfig object to determine which connections to draw
        color: BGR color for drawing
        thickness: line thickness
    """
    # Define all possible connections
    body_connections = [
        # Upper body (always shown if keypoints exist)
        (0, 1), (0, 2),  # nose to eyes
        (1, 3), (2, 4),  # eyes to ears
        (0, 5), (0, 6),  # nose to shoulders
        (5, 6),  # shoulders
        (5, 7), (7, 9),  # left arm
        (6, 8), (8, 10),  # right arm
        (5, 11), (6, 12),  # shoulders to hips
        (11, 12),  # hips
    ]

    leg_connections = [
        # Legs (only if use_legs=True)
        (11, 13), (13, 15),  # left leg
        (12, 14), (14, 16),  # right leg
    ]

    # Build connections list based on config
    connections = body_connections.copy()

    if kp_config is None or kp_config.use_legs:
        connections.extend(leg_connections)

    # Draw connections with different colors based on type
    for start_idx, end_idx in connections:
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            start_point = tuple(keypoints[start_idx].astype(int))
            end_point = tuple(keypoints[end_idx].astype(int))

            # Check if points are valid (not zero)
            if start_point != (0, 0) and end_point != (0, 0):
                # Use different color for legs if specified
                conn_color = color
                if (start_idx, end_idx) in leg_connections:
                    if kp_config and not kp_config.use_legs:
                        continue  # Skip leg connections if legs disabled
                    conn_color = (0, 255, 255)  # Yellow for legs

                cv2.line(frame, start_point, end_point, conn_color, thickness)

    return frame


def draw_keypoints(frame, keypoints, kp_config=None, radius=4):
    """
    Draw keypoints as circles on frame with color coding.

    Args:
        frame: video frame to draw on
        keypoints: array of shape (N, 2) containing x, y coordinates
        kp_config: KeypointConfig object to determine visualization
        radius: circle radius
    """
    for i, (x, y) in enumerate(keypoints):
        point = (int(x), int(y))
        if point == (0, 0):  # Skip invalid points
            continue

        # Determine color and whether to draw based on keypoint type
        color = None
        draw = True

        if i < 13:  # Upper body (always shown)
            color = (0, 0, 255)  # Red
        elif i < 23:  # Legs (13-22)
            if kp_config and not kp_config.use_legs:
                draw = False
            else:
                color = (0, 255, 255)  # Yellow
        elif i < 91:  # Face (23-90)
            if kp_config and not kp_config.use_face:
                draw = False
            else:
                color = (255, 0, 255)  # Magenta
        else:  # Hands (91-132)
            if kp_config and not kp_config.use_hands:
                draw = False
            else:
                color = (255, 165, 0)  # Orange

        if draw and color:
            cv2.circle(frame, point, radius, color, -1)

            # Draw keypoint index for body keypoints
            if i < 23:
                cv2.putText(frame, str(i), point, cv2.FONT_HERSHEY_SIMPLEX,
                           0.3, (255, 255, 255), 1)

    return frame


def visualize_video_with_poses(video_path, keypoints_file, output_path=None,
                               show_connections=True, show_keypoints=True,
                               max_frames=None, kp_config=None):
    """
    Visualize poses on video.

    Args:
        video_path: path to original video
        keypoints_file: path to keypoints pickle file
        output_path: path to save output video (None to just display)
        show_connections: whether to draw skeleton connections
        show_keypoints: whether to draw keypoint circles
        max_frames: maximum number of frames to process (None for all)
        kp_config: KeypointConfig object (None = show all keypoints)
    """
    if kp_config is None:
        kp_config = KeypointConfig(use_legs=True, use_face=True, use_hands=True)
    # Load keypoints
    print(f"Loading keypoints from: {keypoints_file}")
    with open(keypoints_file, 'rb') as f:
        keypoints_data = pickle.load(f)

    # Open video
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = min(len(keypoints_data), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    if max_frames:
        total_frames = min(total_frames, max_frames)

    print(f"Video properties: {frame_width}x{frame_height} @ {fps}fps")
    print(f"Processing {total_frames} frames")

    # Setup video writer if output path provided
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        print(f"Writing output to: {output_path}")

    # Process frames
    frame_idx = 0
    frames_with_pose = 0
    frames_without_pose = 0

    pbar = tqdm(total=total_frames, desc="Processing frames")

    while frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Get keypoints for this frame
        if frame_idx < len(keypoints_data):
            keypoints = keypoints_data[frame_idx]['keypoints']

            if keypoints is not None and len(keypoints) > 0:
                frames_with_pose += 1

                # Draw skeleton connections
                if show_connections:
                    frame = draw_skeleton_connections(frame, keypoints,
                                                     kp_config=kp_config,
                                                     color=(0, 255, 0), thickness=2)

                # Draw keypoints
                if show_keypoints:
                    frame = draw_keypoints(frame, keypoints, kp_config=kp_config, radius=4)

                # Add text showing frame has pose and config
                status_text = f"Frame {frame_idx}: {kp_config.name} ({len(kp_config.keypoint_indices)} kpts)"
                cv2.putText(frame, status_text,
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                           0.6, (0, 255, 0), 2)

                # Add legend
                y_offset = 60
                cv2.putText(frame, "Legend:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20
                cv2.putText(frame, "Body (red)", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                y_offset += 15
                if kp_config.use_legs:
                    cv2.putText(frame, "Legs (yellow)", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    y_offset += 15
                if kp_config.use_face:
                    cv2.putText(frame, "Face (magenta)", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                    y_offset += 15
                if kp_config.use_hands:
                    cv2.putText(frame, "Hands (orange)", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)
            else:
                frames_without_pose += 1
                # Add text showing no pose
                cv2.putText(frame, f"Frame {frame_idx}: No Pose",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (0, 0, 255), 2)

        # Write frame
        if writer:
            writer.write(frame)

        # Display frame (if not writing to file, show in window)
        if not output_path:
            cv2.imshow('Pose Visualization', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nStopped by user")
                break

        frame_idx += 1
        pbar.update(1)

    pbar.close()

    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # Print summary
    print(f"\nSummary:")
    print(f"  Total frames processed: {frame_idx}")
    print(f"  Frames with pose: {frames_with_pose} ({frames_with_pose/frame_idx*100:.1f}%)")
    print(f"  Frames without pose: {frames_without_pose} ({frames_without_pose/frame_idx*100:.1f}%)")


def visualize_random_samples(data_dir, video_root_dir, num_samples=3,
                             output_dir='./visualization_output',
                             kp_config=None):
    """
    Visualize random video samples from each class.

    Args:
        data_dir: path to extracted pose data
        video_root_dir: path to original video dataset
        num_samples: number of random videos per class to visualize
        output_dir: directory to save output videos
        kp_config: KeypointConfig object (None = show all keypoints)
    """
    if kp_config is None:
        kp_config = KeypointConfig(use_legs=True, use_face=True, use_hands=True)
    os.makedirs(output_dir, exist_ok=True)

    # Get all classes
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    classes.sort()

    print(f"Found {len(classes)} classes: {classes}")

    # Process each class
    for class_name in classes:
        print(f"\n{'='*60}")
        print(f"Class: {class_name}")
        print(f"{'='*60}")

        class_pose_dir = os.path.join(data_dir, class_name)
        class_video_dir = os.path.join(video_root_dir, class_name)

        if not os.path.exists(class_video_dir):
            print(f"Warning: Video directory not found: {class_video_dir}")
            continue

        # Get all keypoint files
        keypoint_files = list(Path(class_pose_dir).glob('*_keypoints.pkl'))

        if len(keypoint_files) == 0:
            print(f"No keypoint files found in {class_pose_dir}")
            continue

        # Randomly sample files
        sample_files = random.sample(keypoint_files, min(num_samples, len(keypoint_files)))

        # Process each sample
        for keypoint_file in sample_files:
            # Find corresponding video file
            video_name = keypoint_file.stem.replace('_keypoints', '')

            # Try different video extensions
            video_file = None
            for ext in ['.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV']:
                potential_video = os.path.join(class_video_dir, video_name + ext)
                if os.path.exists(potential_video):
                    video_file = potential_video
                    break

            if not video_file:
                print(f"Warning: Video file not found for {video_name}")
                continue

            # Output path
            config_suffix = kp_config.name.replace("+", "_").replace(" ", "_")
            output_file = os.path.join(output_dir, f"{class_name}_{video_name}_{config_suffix}.mp4")

            # Visualize
            print(f"\nProcessing: {video_name}")
            visualize_video_with_poses(
                video_file,
                keypoint_file,
                output_path=output_file,
                show_connections=True,
                show_keypoints=True,
                max_frames=300,  # Limit to first 300 frames for faster processing
                kp_config=kp_config
            )


def visualize_specific_video(class_name, video_name, data_dir, video_root_dir,
                            output_path=None, display_only=False, kp_config=None):
    """
    Visualize a specific video by class and name.

    Args:
        class_name: name of the class (e.g., 'drinking')
        video_name: name of the video without extension (e.g., 'drinking (1)')
        data_dir: path to extracted pose data
        video_root_dir: path to original video dataset
        output_path: path to save output (None to auto-generate)
        display_only: if True, only display without saving
        kp_config: KeypointConfig object (None = show all keypoints)
    """
    if kp_config is None:
        kp_config = KeypointConfig(use_legs=True, use_face=True, use_hands=True)
    keypoint_file = os.path.join(data_dir, class_name, f"{video_name}_keypoints.pkl")

    # Find video file
    video_file = None
    class_video_dir = os.path.join(video_root_dir, class_name)

    for ext in ['.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV']:
        potential_video = os.path.join(class_video_dir, video_name + ext)
        if os.path.exists(potential_video):
            video_file = potential_video
            break

    if not video_file:
        print(f"Error: Video file not found for {class_name}/{video_name}")
        return

    if not os.path.exists(keypoint_file):
        print(f"Error: Keypoint file not found: {keypoint_file}")
        return

    # Generate output path if needed
    if not display_only and output_path is None:
        config_suffix = kp_config.name.replace("+", "_").replace(" ", "_")
        output_path = f"{class_name}_{video_name}_{config_suffix}.mp4"

    # Visualize
    visualize_video_with_poses(
        video_file,
        keypoint_file,
        output_path=None if display_only else output_path,
        show_connections=True,
        show_keypoints=True,
        kp_config=kp_config
    )


def main():
    """Main function with examples"""

    # Configuration
    data_dir = r'D:\data\EduAction_pose_data'
    video_root_dir = r'D:\data\EduAction-A-college-student-action-dataset-for-classroom-attention-estimation'
    output_dir = r'D:\data\EduAction_visualization'

    print("="*80)
    print("EduAction Pose Visualization")
    print("="*80)
    print("\nKeypoint Configuration Options:")
    print("  1. Full Body     - All keypoints (legs + face + hands)")
    print("  2. Upper Body    - No legs (face + hands only)")
    print("  3. Body + Hands  - No face (legs + hands)")
    print("  4. Hands Only    - Body + hands only (no legs, no face)")
    print("  5. Body Only     - Minimal (no legs, no face, no hands)")
    print("="*80)

    # ==================== CONFIGURATION ====================
    # Choose which keypoint groups to visualize:

    USE_LEGS = True   # Include leg keypoints?
    USE_FACE = True   # Include face keypoints?
    USE_HANDS = True   # Include hand keypoints?

    # =======================================================

    # Create keypoint config
    kp_config = KeypointConfig(use_legs=USE_LEGS, use_face=USE_FACE, use_hands=USE_HANDS)

    print(f"\n>>> Selected Configuration: {kp_config.name}")
    print(f"    Total keypoints: {len(kp_config.keypoint_indices)}")
    print()

    # Option 1: Visualize random samples from each class
    print("Visualizing random samples from each class with selected config...")
    visualize_random_samples(
        data_dir=data_dir,
        video_root_dir=video_root_dir,
        num_samples=1,  # 1 video per class
        output_dir=output_dir,
        kp_config=kp_config
    )

    # Option 2: Visualize specific video with different configs (commented out)
    # print("\nVisualizing same video with different configs for comparison...")
    # configs_to_compare = [
    #     KeypointConfig(use_legs=True, use_face=True, use_hands=True),   # Full
    #     KeypointConfig(use_legs=False, use_face=True, use_hands=True),  # Upper body
    #     KeypointConfig(use_legs=False, use_face=False, use_hands=True), # Hands only
    # ]
    #
    # for config in configs_to_compare:
    #     print(f"\n  Processing with config: {config.name}")
    #     visualize_specific_video(
    #         class_name='drinking',
    #         video_name='drinking (1)',
    #         data_dir=data_dir,
    #         video_root_dir=video_root_dir,
    #         kp_config=config
    #     )

    print("\n" + "="*80)
    print("Visualization complete!")
    print(f"Output saved to: {output_dir}")
    print(f"Config used: {kp_config.name}")
    print("="*80)


if __name__ == "__main__":
    main()
