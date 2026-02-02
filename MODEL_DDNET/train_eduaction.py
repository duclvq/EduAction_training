"""
Training script for EduAction dataset.
This script loads pose data extracted from EduAction videos and trains a DD-Net model.
"""

import os
import sys
import pickle
import random
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from keras.utils import to_categorical
import math

# Import from ddnet module
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ddnet'))
from MODEL_DDNET.ddnet import build_DD_Net, Config
from MODEL_DDNET.utils import zoom, get_CG, sampling_frame

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


class EduActionConfig(Config):
    """Configuration for EduAction dataset training"""

    def __init__(self, use_leg_keypoints=True, use_face_keypoints=True, use_hand_keypoints=True):
        super().__init__()

        # Dataset specific parameters
        self.data_dir = r'D:\data\EduAction_pose_data'

        # EduAction classes (8 classes)
        self.class_names = [
            'drinking',
            'lecture',
            'play_phone',
            'sleeping',
            'talking',
            'watch_computer',
            'writing'
        ]
        self.clc_num = len(self.class_names)

        # Keypoint selection parameters
        self.use_leg_keypoints = use_leg_keypoints
        self.use_face_keypoints = use_face_keypoints
        self.use_hand_keypoints = use_hand_keypoints

        # Wholebody keypoint structure (133 keypoints total):
        # 0-16: Body (COCO format)
        #   0: nose, 1-2: eyes, 3-4: ears
        #   5-6: shoulders, 7-8: elbows, 9-10: wrists
        #   11-12: hips, 13-14: knees, 15-16: ankles
        # 17-22: Feet (6 keypoints)
        # 23-90: Face (68 keypoints)
        # 91-111: Left hand (21 keypoints)
        # 112-132: Right hand (21 keypoints)

        # Build keypoint indices based on configuration
        self.keypoint_indices = []

        # Body keypoints (0-12: upper body without legs)
        if use_leg_keypoints:
            # Include full body (0-16) + feet (17-22)
            self.keypoint_indices.extend(range(0, 23))
        else:
            # Only upper body (0-12: nose, eyes, ears, shoulders, elbows, wrists, hips)
            self.keypoint_indices.extend(range(0, 13))

        # Face keypoints (23-90)
        if use_face_keypoints:
            self.keypoint_indices.extend(range(23, 91))

        # Hand keypoints (91-132)
        if use_hand_keypoints:
            self.keypoint_indices.extend(range(91, 133))

        self.joint_n = len(self.keypoint_indices)

        # Print configuration summary
        config_parts = []
        if use_leg_keypoints:
            config_parts.append("legs")
        if use_face_keypoints:
            config_parts.append("face")
        if use_hand_keypoints:
            config_parts.append("hands")

        config_str = "+".join(config_parts) if config_parts else "body_only"
        print(f"Keypoint configuration: {config_str} → {self.joint_n} keypoints")

        # Model architecture parameters
        self.frame_l = 70  # temporal length
        self.joint_d = 2  # 2D keypoints (x, y)
        # feat_d is the number of pairwise distances between joints
        # For n joints, we have C(n,2) = n*(n-1)/2 pairs
        self.feat_d = self.joint_n * (self.joint_n - 1) // 2  # feature dimension for motion
        self.filters = 60  # base filter size

        # Training parameters
        self.batch_size = 16
        self.epochs = 150
        self.learning_rate = 0.001
        self.min_lr = 1e-6
        self.max_lr = self.learning_rate

        # Data split parameters
        # Note: Train/test split is done randomly with 70-30 ratio (random_state=42)
        # Validation is further split from training data (15% of training set)
        self.test_split_ratio = 0.3  # 30% of all data for testing
        self.validation_split_from_train = 0.15  # 15% of training data for validation

        # Data augmentation
        self.use_data_augmentation = True
        self.augmentation_probability = 0.5

        # Oversampling
        self.apply_oversampling = True
        self.oversampling_strategy = 'balanced'
        self.augmentation_noise_factor = 0.01

        # Learning rate scheduler
        self.use_cosine_scheduler = True
        self.cosine_restarts = False

        # Model saving - generate suffix based on config
        suffix_parts = []
        if use_leg_keypoints:
            suffix_parts.append("legs")
        if use_face_keypoints:
            suffix_parts.append("face")
        if use_hand_keypoints:
            suffix_parts.append("hands")

        if not suffix_parts:
            keypoint_suffix = "body_only"
        else:
            keypoint_suffix = "_".join(suffix_parts)

        self.model_save_path = f'EduAction_DDNet_{keypoint_suffix}.h5'
        self.history_save_path = f'EduAction_training_history_{keypoint_suffix}.pkl'

        # Minimum frames threshold
        self.min_frames = 10  # minimum frames for a valid sequence


def load_eduaction_poses(data_dir, config):
    """
    Load all pose data from EduAction dataset.
    Data will be split into train/test later using train_test_split with fixed seed.

    Args:
        data_dir: path to extracted pose data
        config: EduActionConfig instance

    Returns:
        all_poses: list of pose sequences
        all_labels: list of corresponding labels
        video_metadata: list of metadata for each sample
    """
    all_poses = []
    all_labels = []
    video_metadata = []

    print("Loading EduAction pose data...")
    print(f"Data directory: {data_dir}")

    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} does not exist!")
        print("Please run extract_eduaction_poses.py first to extract pose data.")
        return [], [], []

    # Process each class
    for class_idx, class_name in enumerate(config.class_names):
        class_dir = os.path.join(data_dir, class_name)

        if not os.path.exists(class_dir):
            print(f"Warning: Class directory {class_dir} not found, skipping...")
            continue

        # Find all keypoint files in this class directory
        all_keypoint_files = list(Path(class_dir).glob('*_keypoints.pkl'))

        if len(all_keypoint_files) == 0:
            print(f"Warning: No keypoint files found in {class_dir}")
            continue

        # Use all keypoint files (no filtering by video number)
        keypoint_files = all_keypoint_files

        if len(keypoint_files) == 0:
            print(f"Warning: No files found in {class_dir}")
            continue

        print(f"\nProcessing class: {class_name} ({len(keypoint_files)} videos)")

        valid_count = 0
        skipped_count = 0

        for keypoint_file in tqdm(keypoint_files, desc=f"Loading {class_name}"):
            # Load keypoints
            with open(keypoint_file, 'rb') as f:
                keypoints_data = pickle.load(f)

            # Load metadata
            metadata_file = keypoint_file.parent / keypoint_file.name.replace('_keypoints.pkl', '_metadata.pkl')
            if metadata_file.exists():
                with open(metadata_file, 'rb') as f:
                    metadata = pickle.load(f)
            else:
                metadata = {'video_name': keypoint_file.stem}

            # Extract pose sequence from frames
            pose_sequence = []
            for frame_data in keypoints_data:
                keypoints = frame_data['keypoints']

                # Skip frames with no detected person
                if keypoints is None:
                    continue

                # Filter keypoints based on config (e.g., remove legs)
                if hasattr(config, 'keypoint_indices') and len(config.keypoint_indices) < 133:
                    # Only select specified keypoints
                    if keypoints.shape[0] >= max(config.keypoint_indices) + 1:
                        keypoints = keypoints[config.keypoint_indices, :]

                # Ensure keypoints have correct shape (joint_n, joint_d)
                if keypoints.shape[0] != config.joint_n:
                    # Pad or truncate to match expected number of joints
                    if keypoints.shape[0] < config.joint_n:
                        padding = np.zeros((config.joint_n - keypoints.shape[0], keypoints.shape[1]))
                        keypoints = np.vstack([keypoints, padding])
                    else:
                        keypoints = keypoints[:config.joint_n, :]

                # Ensure correct dimensions (2D)
                if keypoints.shape[1] < config.joint_d:
                    padding = np.zeros((keypoints.shape[0], config.joint_d - keypoints.shape[1]))
                    keypoints = np.hstack([keypoints, padding])
                elif keypoints.shape[1] > config.joint_d:
                    keypoints = keypoints[:, :config.joint_d]

                pose_sequence.append(keypoints)

            # Check if sequence has enough frames
            if len(pose_sequence) < config.min_frames:
                skipped_count += 1
                continue

            # Convert to numpy array: shape (frames, joints, coords)
            pose_array = np.array(pose_sequence)

            all_poses.append(pose_array)
            all_labels.append(class_idx)
            video_metadata.append({
                'class': class_name,
                'class_idx': class_idx,
                'video_name': metadata.get('video_name', 'unknown'),
                'original_frames': len(pose_sequence)
            })

            valid_count += 1

        print(f"  Loaded {valid_count} valid sequences")
        if skipped_count > 0:
            print(f"  Skipped {skipped_count} sequences (less than {config.min_frames} frames)")

    print(f"\n{'='*60}")
    print(f"Total samples loaded: {len(all_poses)}")
    print(f"Class distribution:")
    label_counts = Counter(all_labels)
    for class_idx, count in sorted(label_counts.items()):
        class_name = config.class_names[class_idx]
        percentage = count / len(all_labels) * 100
        print(f"  {class_name}: {count} samples ({percentage:.1f}%)")
    print(f"{'='*60}")

    return all_poses, all_labels, video_metadata


def preprocess_pose_data(poses, labels, config, desc="Processing poses"):
    """
    Preprocess pose sequences for DD-Net training.

    Args:
        poses: list of pose sequences
        labels: list of labels
        config: EduActionConfig instance
        desc: description for progress bar

    Returns:
        X_motion: motion features
        X_pose: normalized pose data
        Y: labels
    """
    X_motion = []
    X_pose = []
    Y = []

    for i, pose in enumerate(tqdm(poses, desc=desc)):
        try:
            # Resize to target frame length
            pose_resized = zoom(pose, target_l=config.frame_l,
                              joints_num=config.joint_n, joints_dim=config.joint_d)

            # Apply data augmentation if enabled
            if config.use_data_augmentation and random.random() < config.augmentation_probability:
                pose_resized = sampling_frame(pose_resized, config)

            # Generate motion features (using get_CG from utils)
            motion_features = get_CG(pose_resized, config)

            X_motion.append(motion_features)
            X_pose.append(pose_resized)
            Y.append(labels[i])

        except Exception as e:
            print(f"\nError processing pose {i}: {e}")
            continue

    # Convert to numpy arrays
    X_motion = np.array(X_motion)
    X_pose = np.array(X_pose)
    Y = np.array(Y)

    print(f"\nPreprocessed data shapes:")
    print(f"  X_motion: {X_motion.shape}")
    print(f"  X_pose: {X_pose.shape}")
    print(f"  Y: {Y.shape}")

    return X_motion, X_pose, Y


def apply_oversampling(X_motion, X_pose, Y, config):
    """
    Apply oversampling to balance class distribution.
    Only applied to training data.
    """
    if not config.apply_oversampling:
        return X_motion, X_pose, Y

    print("\n" + "="*60)
    print("APPLYING OVERSAMPLING")
    print("="*60)

    # Current distribution
    class_counts = Counter(Y)
    print("Current class distribution:")
    for class_idx, count in sorted(class_counts.items()):
        print(f"  {config.class_names[class_idx]}: {count} samples")

    # Target count
    target_count = max(class_counts.values())
    print(f"\nBalancing to {target_count} samples per class")

    # Separate by class
    class_data = {}
    for class_idx in range(config.clc_num):
        mask = Y == class_idx
        class_data[class_idx] = {
            'X_motion': X_motion[mask],
            'X_pose': X_pose[mask],
        }

    # Oversample each class
    oversampled_X_motion = []
    oversampled_X_pose = []
    oversampled_Y = []

    for class_idx in range(config.clc_num):
        current_count = len(class_data[class_idx]['X_motion'])

        # Add original samples
        oversampled_X_motion.extend(class_data[class_idx]['X_motion'])
        oversampled_X_pose.extend(class_data[class_idx]['X_pose'])
        oversampled_Y.extend([class_idx] * current_count)

        # Generate additional samples if needed
        if current_count < target_count:
            samples_needed = target_count - current_count

            for _ in range(samples_needed):
                # Random sample
                idx = random.randint(0, current_count - 1)

                # Add noise for augmentation
                motion = class_data[class_idx]['X_motion'][idx].copy()
                pose = class_data[class_idx]['X_pose'][idx].copy()

                motion += np.random.normal(0, config.augmentation_noise_factor, motion.shape)
                pose += np.random.normal(0, config.augmentation_noise_factor, pose.shape)

                oversampled_X_motion.append(motion)
                oversampled_X_pose.append(pose)
                oversampled_Y.append(class_idx)

    # Convert and shuffle
    oversampled_X_motion = np.array(oversampled_X_motion)
    oversampled_X_pose = np.array(oversampled_X_pose)
    oversampled_Y = np.array(oversampled_Y)

    shuffle_idx = np.random.permutation(len(oversampled_Y))
    oversampled_X_motion = oversampled_X_motion[shuffle_idx]
    oversampled_X_pose = oversampled_X_pose[shuffle_idx]
    oversampled_Y = oversampled_Y[shuffle_idx]

    print("\nOversampling results:")
    print(f"  Original: {len(Y)} samples")
    print(f"  Oversampled: {len(oversampled_Y)} samples")

    return oversampled_X_motion, oversampled_X_pose, oversampled_Y


def cosine_scheduler(epoch, lr, total_epochs, min_lr, max_lr):
    """Cosine annealing learning rate scheduler"""
    return min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))


def plot_training_history(history, save_path='EduAction_training_plots.png'):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy
    ax1.plot(history.history['accuracy'], label='Training')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    # Loss
    ax2.plot(history.history['loss'], label='Training')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training plots saved to {save_path}")
    plt.close()


def evaluate_model(model, X_motion_test, X_pose_test, Y_test_cat, Y_test_raw, config):
    """Evaluate trained model"""
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)

    # Predictions
    predictions = model.predict([X_motion_test, X_pose_test])
    predicted_classes = np.argmax(predictions, axis=1)

    # Accuracy
    test_accuracy = np.mean(predicted_classes == Y_test_raw)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(Y_test_raw, predicted_classes,
                              target_names=config.class_names, digits=4))

    # Confusion matrix
    cm = confusion_matrix(Y_test_raw, predicted_classes)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=config.class_names, yticklabels=config.class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()

    # Save with appropriate name based on config
    suffix_parts = []
    if config.use_leg_keypoints:
        suffix_parts.append("legs")
    if config.use_face_keypoints:
        suffix_parts.append("face")
    if config.use_hand_keypoints:
        suffix_parts.append("hands")

    keypoint_suffix = "_".join(suffix_parts) if suffix_parts else "body_only"
    cm_path = f'EduAction_confusion_matrix_{keypoint_suffix}.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {cm_path}")
    plt.close()

    return test_accuracy, predicted_classes


def train_eduaction_model(use_leg_keypoints=True, use_face_keypoints=True, use_hand_keypoints=True):
    """
    Main training function for EduAction dataset

    Args:
        use_leg_keypoints: If True, include leg keypoints (knees, ankles, feet)
        use_face_keypoints: If True, include face keypoints (68 points)
        use_hand_keypoints: If True, include hand keypoints (42 points total)
    """

    # Initialize configuration
    config = EduActionConfig(
        use_leg_keypoints=use_leg_keypoints,
        use_face_keypoints=use_face_keypoints,
        use_hand_keypoints=use_hand_keypoints
    )

    print("="*60)
    print("EduAction DD-Net Training")
    print("="*60)
    print(f"Configuration:")
    print(f"  Classes: {config.clc_num} ({', '.join(config.class_names)})")
    print(f"  Frame length: {config.frame_l}")
    print(f"  Joints: {config.joint_n}")
    print(f"  Joint dimensions: {config.joint_d}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Learning rate: {config.learning_rate}")
    print("="*60)

    # Load all data
    print("\n" + "="*60)
    print("LOADING ALL DATA")
    print("="*60)
    all_poses, all_labels, all_metadata = load_eduaction_poses(config.data_dir, config)

    if len(all_poses) == 0:
        print("\nError: No data loaded!")
        print("Please check that:")
        print(f"1. Data directory exists: {config.data_dir}")
        print("2. Pose extraction has been completed (run extract_eduaction_poses.py)")
        return

    # Preprocess all data
    print("\n" + "="*60)
    print("PREPROCESSING DATA")
    print("="*60)
    X_motion_all, X_pose_all, Y_all = preprocess_pose_data(
        all_poses, all_labels, config, desc="Processing all data"
    )

    print(f"\nTotal samples loaded: {len(Y_all)}")

    # Print class distribution
    from collections import Counter
    class_counts = Counter(Y_all)
    print("\nClass distribution:")
    for class_idx, count in sorted(class_counts.items()):
        print(f"  {config.class_names[class_idx]}: {count} samples")

    # Split into train (70%) and test (30%) with fixed seed
    print("\n" + "="*60)
    print("SPLITTING DATA: 70% Train / 30% Test (random_state=42)")
    print("="*60)
    X_motion_train_full, X_motion_test, X_pose_train_full, X_pose_test, Y_train_full, Y_test = train_test_split(
        X_motion_all, X_pose_all, Y_all,
        test_size=0.3,
        stratify=Y_all,
        random_state=42
    )

    # Further split training data into train and validation (85% train / 15% val from the 70%)
    print("\nSplitting training data: 85% Train / 15% Validation")
    val_ratio = 0.15  # 15% of training data for validation
    X_motion_train, X_motion_val, X_pose_train, X_pose_val, Y_train, Y_val = train_test_split(
        X_motion_train_full, X_pose_train_full, Y_train_full,
        test_size=val_ratio,
        stratify=Y_train_full,
        random_state=42
    )

    print(f"\nFinal data splits:")
    print(f"  Training: {len(Y_train)} samples (~{len(Y_train)/len(Y_all)*100:.1f}%)")
    print(f"  Validation: {len(Y_val)} samples (~{len(Y_val)/len(Y_all)*100:.1f}%)")
    print(f"  Test: {len(Y_test)} samples (~{len(Y_test)/len(Y_all)*100:.1f}%)")

    # Apply oversampling to training data only
    X_motion_train, X_pose_train, Y_train = apply_oversampling(
        X_motion_train, X_pose_train, Y_train, config
    )

    # Convert labels to categorical
    Y_train_cat = to_categorical(Y_train, num_classes=config.clc_num)
    Y_val_cat = to_categorical(Y_val, num_classes=config.clc_num)
    Y_test_cat = to_categorical(Y_test, num_classes=config.clc_num)

    print(f"\nFinal training samples: {len(Y_train)}")

    # Build model
    print("\nBuilding DD-Net model...")
    print(f"Model config:")
    print(f"  frame_l: {config.frame_l}")
    print(f"  joint_n: {config.joint_n}")
    print(f"  joint_d: {config.joint_d}")
    print(f"  feat_d: {config.feat_d}")
    print(f"  filters: {config.filters}")
    model = build_DD_Net(config)

    # Compile
    model.compile(
        optimizer=Adam(learning_rate=config.learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Model compiled successfully!")
    model.summary()

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            config.model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=30,
            restore_best_weights=True,
            verbose=1
        )
    ]

    # Learning rate scheduler
    if config.use_cosine_scheduler:
        print("Using cosine annealing LR scheduler")
        lr_scheduler = LearningRateScheduler(
            lambda epoch, lr: cosine_scheduler(
                epoch, lr, config.epochs, config.min_lr, config.max_lr
            ),
            verbose=1
        )
        callbacks.append(lr_scheduler)
    else:
        print("Using ReduceLROnPlateau scheduler")
        callbacks.append(
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=config.min_lr,
                verbose=1
            )
        )

    # Train
    print("\nStarting training...")
    history = model.fit(
        [X_motion_train, X_pose_train], Y_train_cat,
        batch_size=config.batch_size,
        epochs=config.epochs,
        validation_data=([X_motion_val, X_pose_val], Y_val_cat),
        callbacks=callbacks,
        verbose=1
    )

    # Save history
    with open(config.history_save_path, 'wb') as f:
        pickle.dump(history.history, f)
    print(f"Training history saved to {config.history_save_path}")

    # Plot training history
    suffix_parts = []
    if config.use_leg_keypoints:
        suffix_parts.append("legs")
    if config.use_face_keypoints:
        suffix_parts.append("face")
    if config.use_hand_keypoints:
        suffix_parts.append("hands")

    keypoint_suffix = "_".join(suffix_parts) if suffix_parts else "body_only"
    plot_training_history(history, save_path=f'EduAction_training_plots_{keypoint_suffix}.png')

    # Load best model
    print("\nLoading best model for evaluation...")
    model.load_weights(config.model_save_path)

    # Evaluate
    test_accuracy, predictions = evaluate_model(
        model, X_motion_test, X_pose_test, Y_test_cat, Y_test, config
    )

    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    print(f"Best model saved to: {config.model_save_path}")
    print(f"Final test accuracy: {test_accuracy:.4f}")
    print("="*60)

    return model, history, test_accuracy


if __name__ == "__main__":
    # GPU configuration
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU available: {len(gpus)} device(s)\n")
        else:
            print("No GPU found, using CPU\n")
    except:
        print("GPU configuration failed, using default settings\n")

    # Training configuration options
    print("="*60)
    print("Training Configuration Options:")
    print("="*60)
    print("Keypoint Groups:")
    print("  - Legs: knees (2) + ankles (2) + feet (6) = 10 keypoints")
    print("  - Face: 68 keypoints")
    print("  - Hands: left (21) + right (21) = 42 keypoints")
    print("  - Body: nose + eyes + ears + shoulders + elbows + wrists + hips = 13 keypoints")
    print("="*60)
    print("Common Configurations:")
    print("  1. Full body: legs + face + hands → 133 keypoints")
    print("  2. Upper body only: face + hands → 123 keypoints")
    print("  3. Body + hands (no face): legs + hands → 65 keypoints")
    print("  4. Body only (minimal): just body → 13-23 keypoints")
    print("="*60)

    # ==================== CONFIGURATION ====================
    # Change these values to experiment with different keypoint combinations:

    USE_LEG_KEYPOINTS = True   # Include legs? (knees, ankles, feet)
    USE_FACE_KEYPOINTS = True  # Include face landmarks? (68 points)
    USE_HAND_KEYPOINTS = True   # Include hand keypoints? (42 points)

    # =======================================================

    # Print selected configuration
    print("\n>>> Selected Configuration:")
    selected = []
    if USE_LEG_KEYPOINTS:
        selected.append("LEGS")
    if USE_FACE_KEYPOINTS:
        selected.append("FACE")
    if USE_HAND_KEYPOINTS:
        selected.append("HANDS")

    if not selected:
        print("    Body only (upper body keypoints)")
    else:
        print(f"    Body + {' + '.join(selected)}")

    print()

    # Run training
    model, history, test_accuracy = train_eduaction_model(
        use_leg_keypoints=USE_LEG_KEYPOINTS,
        use_face_keypoints=USE_FACE_KEYPOINTS,
        use_hand_keypoints=USE_HAND_KEYPOINTS
    )
