"""
Feeder for EduAction dataset with 133 keypoints (COCO-WholeBody format)
Data is stored as individual pickle files per sample in class folders.
Compatible with ST-GCN format: (N, C, T, V, M)
"""
import os
import pickle
import random
import sys
import numpy as np
import torch
from torch.utils.data import Dataset

# Fix numpy compatibility for pickle files saved with numpy 2.x
if not hasattr(np, '_core'):
    import numpy.core as _np_core
    sys.modules['numpy._core'] = _np_core
    sys.modules['numpy._core.multiarray'] = _np_core.multiarray
    sys.modules['numpy._core.numeric'] = _np_core.numeric
    sys.modules['numpy._core._multiarray_umath'] = getattr(_np_core, '_multiarray_umath', _np_core.multiarray)

from . import tools

# Keypoint subset definitions for COCO-WholeBody 133 keypoints
KEYPOINT_SUBSETS = {
    'full_body': list(range(133)),  # All 133 keypoints
    'upper_body': list(range(0, 17)) + list(range(23, 133)),  # No feet (127 keypoints)
    'body_only': list(range(0, 23)),  # Body + feet only (23 keypoints)
    'body_no_feet': list(range(0, 17)),  # Body only, no feet (17 keypoints)
    'face_hands': list(range(23, 133)),  # Face + hands (110 keypoints)
    'hands_only': list(range(91, 133)),  # Both hands (42 keypoints)
    'body_hands': list(range(0, 23)) + list(range(91, 133)),  # Body + hands (65 keypoints)
    'upper_body_hands': list(range(0, 11)) + list(range(91, 133)),  # Upper body + hands (53 keypoints)
    'face_only': list(range(23, 91)),  # Face only (68 keypoints)
}


class FeederEduAction(Dataset):
    """
    Feeder for EduAction skeleton-based action recognition.
    
    Data format: Individual pickle files with keypoints list per video.
    Output format: (C, T, V, M) for ST-GCN compatibility
    
    Args:
        data_dir: Path to EduAction_pose_data folder containing class subfolders
        split: 'train' or 'test'
        random_choose: If True, randomly choose a portion of the input sequence
        random_move: If True, randomly move the skeleton
        window_size: Target number of frames (-1 for no resize)
        debug: If True, only use first 100 samples
        train_ratio: Ratio of data to use for training (same as DDNet: 0.7)
        seed: Random seed for train/test split (same as DDNet: 42)
        keypoint_subset: Which keypoints to use
    """

    def __init__(
        self,
        data_dir,
        split='train',
        random_choose=False,
        random_move=False,
        window_size=-1,
        debug=False,
        train_ratio=0.7,
        seed=42,
        keypoint_subset='full_body',
        mmap=True
    ):
        self.data_dir = data_dir
        self.split = split
        self.random_choose = random_choose
        self.random_move = random_move
        self.window_size = window_size
        self.debug = debug
        self.train_ratio = train_ratio
        self.seed = seed

        # Keypoint selection
        self.keypoint_subset = keypoint_subset
        if keypoint_subset in KEYPOINT_SUBSETS:
            self.selected_keypoints = KEYPOINT_SUBSETS[keypoint_subset]
        elif isinstance(keypoint_subset, list):
            self.selected_keypoints = keypoint_subset
        else:
            print(f"Unknown keypoint_subset: {keypoint_subset}, using full_body")
            self.selected_keypoints = KEYPOINT_SUBSETS['full_body']

        self.num_keypoints = len(self.selected_keypoints)

        # Class names (same as DDNet and MGSAN)
        self.classes = ['drinking', 'lecture', 'play_phone', 'sleeping',
                       'talking', 'watch_computer', 'writing']
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.num_class = len(self.classes)

        self.load_data()

    def load_data(self):
        """Load all samples from class folders and split into train/test."""
        all_samples = []

        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Class directory not found: {class_dir}")
                continue

            # Find all keypoint files
            for f in os.listdir(class_dir):
                if f.endswith('_keypoints.pkl'):
                    sample_path = os.path.join(class_dir, f)
                    all_samples.append({
                        'path': sample_path,
                        'class': class_name,
                        'label': self.class_to_idx[class_name]
                    })

        # Sort for reproducibility
        all_samples.sort(key=lambda x: x['path'])

        # Stratified split (same as DDNet) - ensures balanced class distribution
        from collections import defaultdict
        class_samples = defaultdict(list)
        for sample in all_samples:
            class_samples[sample['label']].append(sample)

        train_samples = []
        test_samples = []

        random.seed(self.seed)
        for label, samples_list in class_samples.items():
            random.shuffle(samples_list)
            n_train = int(len(samples_list) * self.train_ratio)
            train_samples.extend(samples_list[:n_train])
            test_samples.extend(samples_list[n_train:])

        # Shuffle again
        random.shuffle(train_samples)
        random.shuffle(test_samples)

        if self.split == 'train':
            samples = train_samples
        else:
            samples = test_samples

        if self.debug:
            samples = samples[:100]

        # Load all data into memory
        self.data = []
        self.label = []
        self.sample_name = []

        print(f"Loading {len(samples)} samples for {self.split} split...")
        print(f"Using keypoint subset '{self.keypoint_subset}': {self.num_keypoints} keypoints")

        for sample in samples:
            try:
                with open(sample['path'], 'rb') as f:
                    keypoints_list = pickle.load(f)

                # Convert list of dicts to numpy array: (T, V, C)
                frames = []
                for frame_data in keypoints_list:
                    if isinstance(frame_data, dict):
                        kp = frame_data['keypoints']  # (133, 2)
                    else:
                        kp = frame_data  # Already numpy array

                    # Select only the desired keypoints
                    kp = kp[self.selected_keypoints, :]
                    frames.append(kp)

                data = np.array(frames)  # (T, V, C) = (T, num_keypoints, 2)
                
                # Convert to ST-GCN format: (C, T, V, M)
                T, V, C = data.shape
                data = data.transpose(2, 0, 1)  # (C, T, V)
                data = np.expand_dims(data, axis=-1)  # (C, T, V, 1) for M=1

                self.data.append(data)
                self.label.append(sample['label'])
                self.sample_name.append(os.path.basename(sample['path']))

            except Exception as e:
                print(f"Error loading {sample['path']}: {e}")
                continue

        # Get max frames for padding info
        self.max_frames = max(d.shape[1] for d in self.data) if self.data else 0
        print(f"Loaded {len(self.data)} samples successfully. Max frames: {self.max_frames}")

        # Store shapes
        if len(self.data) > 0:
            self.C = self.data[0].shape[0]  # 2 for x, y
            self.V = self.data[0].shape[2]  # num_keypoints
            self.M = self.data[0].shape[3]  # 1 for single person

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # Get data: (C, T, V, M)
        data_numpy = np.array(self.data[index]).astype(np.float32)
        label = self.label[index]

        # Apply data augmentation
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
            
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        return data_numpy, label

    def get_class_weights(self):
        """Calculate class weights for imbalanced dataset."""
        class_counts = np.bincount(self.label, minlength=self.num_class)
        total = len(self.label)
        weights = total / (self.num_class * class_counts + 1e-6)
        return torch.FloatTensor(weights)


class FeederEduActionNpy(Dataset):
    """
    Alternative feeder that loads pre-processed .npy files.
    Compatible with standard ST-GCN data format.
    
    Args:
        data_path: Path to .npy data file, shape (N, C, T, V, M)
        label_path: Path to label .pkl file
        random_choose: If True, randomly choose a portion of the input sequence
        random_move: If True, randomly move the skeleton
        window_size: Target number of frames
        debug: If True, only use first 100 samples
        data_ratio: Fraction of data to use
    """

    def __init__(
        self,
        data_path,
        label_path,
        random_choose=False,
        random_move=False,
        window_size=-1,
        debug=False,
        data_ratio=1.0,
        mmap=True
    ):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_move = random_move
        self.window_size = window_size
        self.data_ratio = data_ratio

        self.load_data(mmap)

    def load_data(self, mmap):
        # Load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        # Load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        # Apply data ratio
        if 0 < self.data_ratio < 1.0:
            num_samples = int(len(self.label) * self.data_ratio)
            indices = np.random.choice(len(self.label), num_samples, replace=False)
            self.label = [self.label[i] for i in indices]
            self.data = self.data[indices]
            self.sample_name = [self.sample_name[i] for i in indices]

        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

        self.N, self.C, self.T, self.V, self.M = self.data.shape

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_numpy = np.array(self.data[index]).astype(np.float32)
        label = self.label[index]

        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        return data_numpy, label
