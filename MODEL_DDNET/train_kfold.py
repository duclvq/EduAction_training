#!/usr/bin/env python
"""
K-Fold Cross Validation Training Script for DD-Net on EduAction dataset.

Features:
- Stratified K-Fold with deterministic splitting
- All random seeds fixed for reproducibility (Python, NumPy, TensorFlow)
- Automatic result aggregation across folds
- Confusion matrix and per-class accuracy tracking

Usage:
    python train_kfold.py --n_folds 5 --seed 42 --keypoint_config full_body
    python train_kfold.py --n_folds 5 --seed 42 --keypoint_config upper_body
"""

import os
import sys
import argparse
import json
import pickle
import random
from collections import Counter, defaultdict
from datetime import datetime
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.utils import to_categorical
import math

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MODEL_DDNET.ddnet import build_DD_Net
from MODEL_DDNET.utils import zoom, get_CG


def set_all_seeds(seed):
    """Set all random seeds for complete reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # TensorFlow deterministic operations
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'


# Keypoint configurations matching ST-GCN
KEYPOINT_CONFIGS = {
    'full_body': {
        'indices': list(range(133)),
        'name': 'Full Body (133 keypoints)',
    },
    'upper_body': {
        'indices': list(range(0, 17)) + list(range(23, 133)),
        'name': 'Upper Body (127 keypoints)',
    },
    'body_only': {
        'indices': list(range(0, 23)),
        'name': 'Body Only (23 keypoints)',
    },
    'body_hands': {
        'indices': list(range(0, 23)) + list(range(91, 133)),
        'name': 'Body + Hands (65 keypoints)',
    },
    'hands_only': {
        'indices': list(range(91, 133)),
        'name': 'Hands Only (42 keypoints)',
    },
    'face_hands': {
        'indices': list(range(23, 133)),
        'name': 'Face + Hands (110 keypoints)',
    },
}


class Config:
    """Configuration class for DD-Net."""
    def __init__(self, keypoint_config='full_body'):
        self.keypoint_config = keypoint_config
        kp_cfg = KEYPOINT_CONFIGS[keypoint_config]

        self.keypoint_indices = kp_cfg['indices']
        self.joint_n = len(self.keypoint_indices)
        self.joint_d = 2
        self.frame_l = 64
        self.feat_d = self.joint_n * (self.joint_n - 1) // 2
        self.filters = 60
        self.clc_num = 7

        self.class_names = ['drinking', 'lecture', 'play_phone', 'sleeping',
                           'talking', 'watch_computer', 'writing']


def load_all_data(data_dir, config):
    """Load all samples from data directory."""
    all_poses = []
    all_labels = []
    all_paths = []

    print(f"Loading data from: {data_dir}")
    print(f"Using keypoint config: {config.keypoint_config} ({config.joint_n} keypoints)")

    for class_idx, class_name in enumerate(config.class_names):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: {class_dir} not found")
            continue

        pkl_files = sorted([f for f in os.listdir(class_dir) if f.endswith('_keypoints.pkl')])

        for pkl_file in pkl_files:
            pkl_path = os.path.join(class_dir, pkl_file)
            try:
                with open(pkl_path, 'rb') as f:
                    keypoints_list = pickle.load(f)

                frames = []
                for frame_data in keypoints_list:
                    if isinstance(frame_data, dict):
                        kp = frame_data['keypoints']
                    else:
                        kp = frame_data

                    # Select keypoints
                    kp = kp[config.keypoint_indices, :config.joint_d]
                    frames.append(kp)

                pose = np.array(frames)  # (T, V, C)

                if len(pose) >= 10:  # Minimum frames
                    all_poses.append(pose)
                    all_labels.append(class_idx)
                    all_paths.append(pkl_path)

            except Exception as e:
                print(f"Error loading {pkl_path}: {e}")
                continue

    print(f"Loaded {len(all_poses)} samples")
    return all_poses, all_labels, all_paths


def preprocess_sample(pose, config, rng=None):
    """Preprocess a single sample with optional seeded RNG."""
    # Resize to target frame length
    pose_resized = zoom(pose, target_l=config.frame_l,
                       joints_num=config.joint_n, joints_dim=config.joint_d)

    # Generate motion features
    motion_features = get_CG(pose_resized, config)

    return motion_features, pose_resized


def create_stratified_kfold_indices(labels, n_folds, seed):
    """Create stratified K-Fold indices with deterministic seed."""
    rng = random.Random(seed)

    # Group by class
    class_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)

    # Shuffle each class with seed
    for label in class_indices:
        rng.shuffle(class_indices[label])

    # Create folds
    folds = [[] for _ in range(n_folds)]

    for label in sorted(class_indices.keys()):
        indices = class_indices[label]
        n_samples = len(indices)
        fold_size = n_samples // n_folds
        remainder = n_samples % n_folds

        start = 0
        for fold_idx in range(n_folds):
            size = fold_size + (1 if fold_idx < remainder else 0)
            folds[fold_idx].extend(indices[start:start + size])
            start += size

    # Shuffle each fold
    for fold_idx in range(n_folds):
        rng.shuffle(folds[fold_idx])

    return folds


def cosine_scheduler(epoch, lr, total_epochs, min_lr, max_lr):
    """Cosine annealing learning rate scheduler."""
    return min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))


class KFoldTrainer:
    """K-Fold Cross Validation Trainer for DD-Net."""

    def __init__(self, args):
        self.args = args
        self.config = Config(args.keypoint_config)

        # Set seeds
        set_all_seeds(args.seed)

        # Create work directory
        self.work_dir = os.path.join(
            args.work_dir,
            f'ddnet_kfold_{args.keypoint_config}'
        )
        os.makedirs(self.work_dir, exist_ok=True)

        # Save config
        config_dict = {
            'n_folds': args.n_folds,
            'seed': args.seed,
            'keypoint_config': args.keypoint_config,
            'joint_n': self.config.joint_n,
            'frame_l': self.config.frame_l,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
        }
        with open(os.path.join(self.work_dir, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)

    def preprocess_fold_data(self, poses, labels, indices, fold_seed, is_train=True):
        """Preprocess data for a fold with deterministic augmentation."""
        X_motion = []
        X_pose = []
        Y = []

        rng = np.random.RandomState(fold_seed)

        for i, idx in enumerate(indices):
            pose = poses[idx]
            label = labels[idx]

            # Use sample-specific seed for reproducibility
            sample_seed = fold_seed + i
            sample_rng = np.random.RandomState(sample_seed)

            try:
                motion, pose_resized = preprocess_sample(pose, self.config, sample_rng)
                X_motion.append(motion)
                X_pose.append(pose_resized)
                Y.append(label)
            except Exception as e:
                print(f"Error preprocessing sample {idx}: {e}")
                continue

        return np.array(X_motion), np.array(X_pose), np.array(Y)

    def train_fold(self, fold_idx, train_indices, val_indices, poses, labels, log_file):
        """Train a single fold."""
        print(f"\n{'='*60}")
        print(f"Training Fold {fold_idx + 1}/{self.args.n_folds}")
        print(f"{'='*60}")

        # Set seed for this fold
        fold_seed = self.args.seed + fold_idx * 1000
        set_all_seeds(fold_seed)

        # Preprocess data
        print("Preprocessing training data...")
        X_motion_train, X_pose_train, Y_train = self.preprocess_fold_data(
            poses, labels, train_indices, fold_seed, is_train=True
        )

        print("Preprocessing validation data...")
        X_motion_val, X_pose_val, Y_val = self.preprocess_fold_data(
            poses, labels, val_indices, fold_seed + 500000, is_train=False
        )

        print(f"Train samples: {len(Y_train)}, Val samples: {len(Y_val)}")

        # Convert to categorical
        Y_train_cat = to_categorical(Y_train, num_classes=self.config.clc_num)
        Y_val_cat = to_categorical(Y_val, num_classes=self.config.clc_num)

        # Build model with deterministic initialization
        tf.keras.utils.set_random_seed(fold_seed)
        model = build_DD_Net(self.config)

        model.compile(
            optimizer=Adam(learning_rate=self.args.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Create fold directory
        fold_dir = os.path.join(self.work_dir, f'fold_{fold_idx + 1}')
        os.makedirs(fold_dir, exist_ok=True)

        # Callbacks
        callbacks = [
            ModelCheckpoint(
                os.path.join(fold_dir, 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=0
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=30,
                restore_best_weights=True,
                verbose=0
            ),
            LearningRateScheduler(
                lambda epoch, lr: cosine_scheduler(
                    epoch, lr, self.args.epochs, 1e-6, self.args.learning_rate
                ),
                verbose=0
            )
        ]

        # Train
        history = model.fit(
            [X_motion_train, X_pose_train], Y_train_cat,
            batch_size=self.args.batch_size,
            epochs=self.args.epochs,
            validation_data=([X_motion_val, X_pose_val], Y_val_cat),
            callbacks=callbacks,
            verbose=1
        )

        # Save history
        with open(os.path.join(fold_dir, 'history.json'), 'w') as f:
            json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)

        # Load best model and evaluate
        model.load_weights(os.path.join(fold_dir, 'best_model.h5'))

        predictions = model.predict([X_motion_val, X_pose_val])
        predicted_classes = np.argmax(predictions, axis=1)

        # Calculate metrics
        accuracy = np.mean(predicted_classes == Y_val)
        cm = confusion_matrix(Y_val, predicted_classes)

        per_class_acc = []
        for i in range(self.config.clc_num):
            if cm[i].sum() > 0:
                per_class_acc.append(cm[i][i] / cm[i].sum())
            else:
                per_class_acc.append(0.0)

        # Log results
        log_msg = f"[Fold {fold_idx + 1}] Best Val Accuracy: {accuracy*100:.2f}%"
        print(log_msg)
        log_file.write(log_msg + '\n')
        log_file.flush()

        # Save fold result
        fold_result = {
            'fold_idx': fold_idx,
            'best_val_acc': float(accuracy * 100),
            'best_epoch': int(np.argmax(history.history['val_accuracy']) + 1),
            'confusion_matrix': cm.tolist(),
            'per_class_acc': per_class_acc,
        }

        with open(os.path.join(fold_dir, 'result.json'), 'w') as f:
            json.dump(fold_result, f, indent=2)

        # Clear memory
        del model
        tf.keras.backend.clear_session()

        return fold_result

    def run(self):
        """Run K-Fold cross validation."""
        print(f"\n{'#'*60}")
        print(f"K-Fold Cross Validation Training - DD-Net")
        print(f"{'#'*60}")
        print(f"Work dir: {self.work_dir}")
        print(f"Folds: {self.args.n_folds}")
        print(f"Seed: {self.args.seed}")
        print(f"Keypoints: {self.config.keypoint_config} ({self.config.joint_n})")

        # Load all data
        poses, labels, paths = load_all_data(self.args.data_dir, self.config)

        if len(poses) == 0:
            print("No data loaded!")
            return None

        # Create K-Fold indices
        fold_indices = create_stratified_kfold_indices(labels, self.args.n_folds, self.args.seed)

        # Open log file
        log_path = os.path.join(self.work_dir, 'training_log.txt')
        log_file = open(log_path, 'w')
        log_file.write(f"Training started: {datetime.now()}\n\n")

        # Train all folds
        all_results = []
        start_time = time.time()

        for fold_idx in range(self.args.n_folds):
            # Get train/val indices
            val_indices = fold_indices[fold_idx]
            train_indices = []
            for i in range(self.args.n_folds):
                if i != fold_idx:
                    train_indices.extend(fold_indices[i])

            fold_result = self.train_fold(
                fold_idx, train_indices, val_indices, poses, labels, log_file
            )
            all_results.append(fold_result)

        total_time = time.time() - start_time

        # Aggregate results
        all_val_accs = [r['best_val_acc'] for r in all_results]
        mean_acc = np.mean(all_val_accs)
        std_acc = np.std(all_val_accs)

        # Aggregate confusion matrices
        total_cm = np.zeros((self.config.clc_num, self.config.clc_num), dtype=int)
        for r in all_results:
            total_cm += np.array(r['confusion_matrix'])

        # Average per-class accuracy
        avg_per_class_acc = np.mean([r['per_class_acc'] for r in all_results], axis=0)

        # Print summary
        summary = f"""
{'='*60}
K-FOLD CROSS VALIDATION SUMMARY - DD-Net
{'='*60}
Folds: {self.args.n_folds}
Seed: {self.args.seed}
Keypoint config: {self.config.keypoint_config} ({self.config.joint_n} keypoints)
Total training time: {total_time/60:.1f} minutes

FOLD RESULTS:
"""
        for i, r in enumerate(all_results):
            summary += f"  Fold {i + 1}: {r['best_val_acc']:.2f}% (epoch {r['best_epoch']})\n"

        summary += f"""
OVERALL:
  Mean Accuracy: {mean_acc:.2f}% (+/- {std_acc:.2f}%)
  Min: {min(all_val_accs):.2f}%
  Max: {max(all_val_accs):.2f}%

PER-CLASS ACCURACY (averaged across folds):
"""
        for i, (cls, acc) in enumerate(zip(self.config.class_names, avg_per_class_acc)):
            summary += f"  {cls:20s}: {acc*100:.2f}%\n"

        summary += f"\n{'='*60}\n"

        print(summary)
        log_file.write(summary)
        log_file.close()

        # Save final summary
        final_summary = {
            'n_folds': self.args.n_folds,
            'seed': self.args.seed,
            'keypoint_config': self.config.keypoint_config,
            'joint_n': self.config.joint_n,
            'mean_accuracy': float(mean_acc),
            'std_accuracy': float(std_acc),
            'fold_results': all_results,
            'aggregated_confusion_matrix': total_cm.tolist(),
            'avg_per_class_accuracy': [float(a) for a in avg_per_class_acc],
            'total_training_time_seconds': total_time,
        }

        with open(os.path.join(self.work_dir, 'kfold_summary.json'), 'w') as f:
            json.dump(final_summary, f, indent=2)

        # Plot aggregated confusion matrix
        plt.figure(figsize=(10, 8))
        cm_norm = total_cm.astype('float') / total_cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                   xticklabels=self.config.class_names,
                   yticklabels=self.config.class_names)
        plt.title(f'DD-Net K-Fold Aggregated Confusion Matrix\n{self.config.keypoint_config} ({mean_acc:.2f}% +/- {std_acc:.2f}%)')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(self.work_dir, 'confusion_matrix.png'), dpi=300)
        plt.close()

        print(f"\nResults saved to: {self.work_dir}")

        return final_summary


def main():
    parser = argparse.ArgumentParser(description='K-Fold Cross Validation for DD-Net')
    parser.add_argument('--data_dir', type=str, default='/mnt/d/data/EduAction_pose_data',
                       help='Path to EduAction pose data')
    parser.add_argument('--work_dir', type=str, default='./work_dir',
                       help='Work directory for saving results')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--keypoint_config', type=str, default='full_body',
                       choices=list(KEYPOINT_CONFIGS.keys()),
                       help='Keypoint configuration')
    parser.add_argument('--epochs', type=int, default=120, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')

    args = parser.parse_args()

    # GPU configuration
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU available: {len(gpus)} device(s)")
    except Exception as e:
        print(f"GPU config failed: {e}")

    trainer = KFoldTrainer(args)
    trainer.run()


if __name__ == '__main__':
    main()
