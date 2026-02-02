"""
Feeder for EduAction dataset with 133 keypoints (COCO-WholeBody format)
Data is stored as individual pickle files per sample in class folders.
"""
import os
import pickle
import random
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

# Fix numpy compatibility for pickle files saved with numpy 2.x
if not hasattr(np, '_core'):
    import numpy.core as _np_core
    sys.modules['numpy._core'] = _np_core
    sys.modules['numpy._core.multiarray'] = _np_core.multiarray
    sys.modules['numpy._core.numeric'] = _np_core.numeric
    sys.modules['numpy._core._multiarray_umath'] = getattr(_np_core, '_multiarray_umath', _np_core.multiarray)


class FeederEduAction(Dataset):
    def __init__(
        self,
        data_dir,
        split='train',
        window_size=64,
        p_interval=[0.5, 1],
        random_rot=False,
        debug=False,
        train_ratio=0.8,
        seed=42,
        bone=False,
        vel=False,
    ):
        """
        EduAction dataset feeder.

        Args:
            data_dir: Path to EduAction_pose_data folder containing class subfolders
            split: 'train' or 'test'
            window_size: Target number of frames to resize to
            p_interval: Cropping interval for data augmentation
            random_rot: Whether to apply random rotation
            debug: If True, only load first 100 samples
            train_ratio: Ratio of data to use for training
            seed: Random seed for train/test split
            bone: Whether to use bone modality
            vel: Whether to use velocity modality
        """
        self.data_dir = data_dir
        self.split = split
        self.window_size = window_size
        self.p_interval = p_interval if isinstance(p_interval, list) else [p_interval]
        self.random_rot = random_rot
        self.debug = debug
        self.train_ratio = train_ratio
        self.seed = seed
        self.bone = bone
        self.vel = vel

        # Class names
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

        # Split into train/test
        random.seed(self.seed)
        random.shuffle(all_samples)

        n_train = int(len(all_samples) * self.train_ratio)

        if self.split == 'train':
            samples = all_samples[:n_train]
        else:
            samples = all_samples[n_train:]

        if self.debug:
            samples = samples[:100]

        # Load all data into memory
        self.data = []
        self.label = []
        self.sample_name = []

        print(f"Loading {len(samples)} samples for {self.split}...")
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
                    frames.append(kp)

                data = np.array(frames)  # (T, V, C) = (T, 133, 2)

                self.data.append(data)
                self.label.append(sample['label'])
                self.sample_name.append(os.path.basename(sample['path']))

            except Exception as e:
                print(f"Error loading {sample['path']}: {e}")
                continue

        print(f"Loaded {len(self.data)} samples successfully")

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # Get data: (T, V, C) where V=133, C=2
        data_numpy = self.data[index].copy()
        label = self.label[index]

        T, V, C = data_numpy.shape

        # Convert to (C, T, V, M) format expected by tools
        # M=1 for single person
        data_numpy = data_numpy.transpose(2, 0, 1)  # (C, T, V)
        data_numpy = np.expand_dims(data_numpy, -1)  # (C, T, V, 1)

        # Calculate valid frames (non-zero)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        if valid_frame_num == 0:
            valid_frame_num = T

        # Apply cropping and resizing
        data_numpy = self.valid_crop_resize(
            data_numpy, valid_frame_num, self.p_interval, self.window_size
        )

        # Apply random rotation for training
        if self.random_rot and self.split == 'train':
            data_numpy = self.random_rotation_2d(data_numpy)

        # Bone modality
        if self.bone:
            data_numpy = self.get_bone_data(data_numpy)

        # Velocity modality
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        return torch.tensor(data_numpy, dtype=torch.float32), label, index

    def valid_crop_resize(self, data_numpy, valid_frame_num, p_interval, window):
        """Crop and resize data to target window size."""
        C, T, V, M = data_numpy.shape
        begin = 0
        end = valid_frame_num
        valid_size = end - begin

        if valid_size == 0:
            valid_size = T
            end = T

        # Crop
        if len(p_interval) == 1:
            p = p_interval[0]
            bias = int((1 - p) * valid_size / 2)
            data = data_numpy[:, begin + bias:end - bias, :, :]
            cropped_length = data.shape[1]
        else:
            p = np.random.rand() * (p_interval[1] - p_interval[0]) + p_interval[0]
            cropped_length = min(max(int(np.floor(valid_size * p)), 16), valid_size)
            bias = np.random.randint(0, max(valid_size - cropped_length + 1, 1))
            data = data_numpy[:, begin + bias:begin + bias + cropped_length, :, :]

        if data.shape[1] == 0:
            data = data_numpy[:, :window, :, :]
            cropped_length = data.shape[1]

        # Resize using interpolation
        data = torch.tensor(data, dtype=torch.float)
        data = data.permute(0, 2, 3, 1).contiguous().view(C * V * M, cropped_length)
        data = data[None, None, :, :]
        data = F.interpolate(
            data, size=(C * V * M, window),
            mode='bilinear', align_corners=False
        ).squeeze()
        data = data.contiguous().view(C, V, M, window).permute(0, 3, 1, 2).contiguous().numpy()

        return data

    def random_rotation_2d(self, data_numpy, theta=0.3):
        """Apply random 2D rotation."""
        C, T, V, M = data_numpy.shape

        angle = np.random.uniform(-theta, theta)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        # Rotation matrix for 2D
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

        # Apply rotation
        for t in range(T):
            for m in range(M):
                xy = data_numpy[:, t, :, m]  # (2, V)
                data_numpy[:, t, :, m] = rot @ xy

        return data_numpy

    def get_bone_data(self, data_numpy):
        """Convert joint data to bone data using skeleton connections."""
        # Define bone pairs for COCO-WholeBody 133 keypoints
        bone_pairs = self.get_bone_pairs()

        C, T, V, M = data_numpy.shape
        bone_data = np.zeros_like(data_numpy)

        for v1, v2 in bone_pairs:
            if v1 < V and v2 < V:
                bone_data[:, :, v1, :] = data_numpy[:, :, v1, :] - data_numpy[:, :, v2, :]

        return bone_data

    def get_bone_pairs(self):
        """Define bone pairs for 133 keypoints skeleton."""
        # Body connections
        pairs = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 0), (6, 0),  # Shoulders to nose
            (7, 5), (9, 7),  # Left arm
            (8, 6), (10, 8),  # Right arm
            (11, 5), (12, 6),  # Hips to shoulders
            (13, 11), (15, 13),  # Left leg
            (14, 12), (16, 14),  # Right leg
        ]

        # Feet
        pairs += [(17, 15), (18, 15), (19, 15)]  # Left foot
        pairs += [(20, 16), (21, 16), (22, 16)]  # Right foot

        # Hands (simplified - connect to wrist)
        for i in range(91, 112):  # Left hand
            pairs.append((i, 91 if i != 91 else 9))
        for i in range(112, 133):  # Right hand
            pairs.append((i, 112 if i != 112 else 10))

        return pairs

    def top_k(self, score, top_k):
        """Calculate top-k accuracy."""
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)
