"""
K-Fold Cross Validation Feeder for EduAction dataset - MGSAN.
Supports stratified K-Fold with deterministic random seeds for reproducibility.
All random operations use seeded generators for exact reproducibility across runs.
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


# Keypoint subset definitions for COCO-WholeBody 133 keypoints
KEYPOINT_SUBSETS = {
    'full_body': list(range(133)),
    'upper_body': list(range(0, 17)) + list(range(23, 133)),
    'body_only': list(range(0, 23)),
    'body_no_feet': list(range(0, 17)),
    'face_hands': list(range(23, 133)),
    'hands_only': list(range(91, 133)),
    'body_hands': list(range(0, 23)) + list(range(91, 133)),
    'upper_body_hands': list(range(0, 11)) + list(range(91, 133)),
    'face_only': list(range(23, 91)),
    'coco17': list(range(0, 17)),  # 17 COCO body joints (ViPose extraction)
}


def set_all_seeds(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class FeederEduActionKFold(Dataset):
    """
    K-Fold Cross Validation Feeder for EduAction - MGSAN.

    Implements stratified K-Fold splitting with deterministic randomness.
    All random operations use seeded generators for exact reproducibility.

    Args:
        data_dir: Path to EduAction_pose_data folder
        split: 'train' or 'val' (validation fold)
        n_folds: Number of folds (default: 5)
        fold_idx: Current fold index (0 to n_folds-1)
        window_size: Target number of frames
        p_interval: Cropping interval for augmentation
        random_rot: Apply random rotation (deterministic with seed)
        debug: If True, only use first 100 samples
        seed: Base random seed for fold splitting
        keypoint_subset: Which keypoints to use
        augment_seed: Seed for data augmentation
        bone: Use bone modality
        vel: Use velocity modality
    """

    def __init__(
        self,
        data_dir,
        split='train',
        n_folds=5,
        fold_idx=0,
        window_size=64,
        p_interval=[0.5, 1],
        random_rot=False,
        debug=False,
        seed=42,
        keypoint_subset='full_body',
        augment_seed=None,
        bone=False,
        vel=False,
    ):
        self.data_dir = data_dir
        self.split = split
        self.n_folds = n_folds
        self.fold_idx = fold_idx
        self.window_size = window_size
        self.p_interval = p_interval if isinstance(p_interval, list) else [p_interval]
        self.random_rot = random_rot
        self.debug = debug
        self.seed = seed
        self.augment_seed = augment_seed if augment_seed is not None else seed
        self.bone = bone
        self.vel = vel

        # Create dedicated random generators for reproducibility
        self.split_rng = random.Random(seed)

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

        # Class names
        self.classes = ['drinking', 'lecture', 'play_phone', 'sleeping',
                       'talking', 'watch_computer', 'writing']
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.num_class = len(self.classes)

        self.load_data()

    def set_augment_seed(self, seed):
        """Update augmentation seed (call at each epoch for varied but reproducible augmentation)."""
        self.augment_seed = seed

    def load_data(self):
        """Load data and perform stratified K-Fold split."""
        all_samples = []

        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Class directory not found: {class_dir}")
                continue

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

        # Stratified K-Fold split
        from collections import defaultdict
        class_samples = defaultdict(list)
        for sample in all_samples:
            class_samples[sample['label']].append(sample)

        train_samples = []
        val_samples = []

        # Use dedicated RNG for splitting
        for label in sorted(class_samples.keys()):
            samples_list = class_samples[label].copy()
            # Shuffle with dedicated RNG
            self.split_rng.shuffle(samples_list)

            n_samples = len(samples_list)
            fold_size = n_samples // self.n_folds
            remainder = n_samples % self.n_folds

            # Calculate fold boundaries
            fold_sizes = [fold_size + (1 if i < remainder else 0) for i in range(self.n_folds)]
            fold_starts = [sum(fold_sizes[:i]) for i in range(self.n_folds)]

            # Get validation indices for current fold
            val_start = fold_starts[self.fold_idx]
            val_end = val_start + fold_sizes[self.fold_idx]

            for i, sample in enumerate(samples_list):
                if val_start <= i < val_end:
                    val_samples.append(sample)
                else:
                    train_samples.append(sample)

        # Shuffle samples (with dedicated RNG)
        train_rng = random.Random(self.seed + 1000)
        val_rng = random.Random(self.seed + 2000)
        train_rng.shuffle(train_samples)
        val_rng.shuffle(val_samples)

        if self.split == 'train':
            samples = train_samples
        else:
            samples = val_samples

        if self.debug:
            samples = samples[:100]

        # Load all data into memory
        self.data = []
        self.label = []
        self.sample_name = []

        print(f"[Fold {self.fold_idx + 1}/{self.n_folds}] Loading {len(samples)} samples for {self.split}...")
        print(f"Using keypoint subset '{self.keypoint_subset}': {self.num_keypoints} keypoints")

        for sample in samples:
            try:
                with open(sample['path'], 'rb') as f:
                    keypoints_list = pickle.load(f)

                frames = []
                for frame_data in keypoints_list:
                    if isinstance(frame_data, dict):
                        kp = frame_data['keypoints']
                    else:
                        kp = frame_data

                    kp = kp[self.selected_keypoints, :]
                    frames.append(kp)

                data = np.array(frames)  # (T, V, C)

                self.data.append(data)
                self.label.append(sample['label'])
                self.sample_name.append(os.path.basename(sample['path']))

            except Exception as e:
                print(f"Error loading {sample['path']}: {e}")
                continue

        print(f"Loaded {len(self.data)} samples")

        # Print class distribution
        class_counts = np.bincount(self.label, minlength=self.num_class)
        print(f"Class distribution: {dict(zip(self.classes, class_counts))}")

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # Get data: (T, V, C)
        data_numpy = self.data[index].copy()
        label = self.label[index]

        T, V, C = data_numpy.shape

        # Convert to (C, T, V, M) format
        data_numpy = data_numpy.transpose(2, 0, 1)  # (C, T, V)
        data_numpy = np.expand_dims(data_numpy, -1)  # (C, T, V, 1)

        # Calculate valid frames
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        if valid_frame_num == 0:
            valid_frame_num = T

        # Apply deterministic cropping and resizing
        data_numpy = self._valid_crop_resize_deterministic(
            data_numpy, valid_frame_num, index
        )

        # Apply deterministic random rotation for training
        if self.random_rot and self.split == 'train':
            data_numpy = self._random_rotation_deterministic(data_numpy, index)

        # Bone modality
        if self.bone:
            data_numpy = self._get_bone_data(data_numpy)

        # Velocity modality
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        return torch.tensor(data_numpy, dtype=torch.float32), label, index

    def _valid_crop_resize_deterministic(self, data_numpy, valid_frame_num, index):
        """Deterministic crop and resize using seeded RNG."""
        C, T, V, M = data_numpy.shape
        begin = 0
        end = valid_frame_num
        valid_size = end - begin

        if valid_size == 0:
            valid_size = T
            end = T

        # Create sample-specific RNG
        rng = np.random.RandomState(self.augment_seed + index)

        # Crop
        if len(self.p_interval) == 1 or self.split != 'train':
            p = self.p_interval[0] if len(self.p_interval) == 1 else 1.0
            bias = int((1 - p) * valid_size / 2)
            data = data_numpy[:, begin + bias:end - bias, :, :]
            cropped_length = data.shape[1]
        else:
            p = rng.rand() * (self.p_interval[1] - self.p_interval[0]) + self.p_interval[0]
            cropped_length = min(max(int(np.floor(valid_size * p)), 16), valid_size)
            bias = rng.randint(0, max(valid_size - cropped_length + 1, 1))
            data = data_numpy[:, begin + bias:begin + bias + cropped_length, :, :]

        if data.shape[1] == 0:
            data = data_numpy[:, :self.window_size, :, :]
            cropped_length = data.shape[1]

        # Resize using interpolation
        data = torch.tensor(data, dtype=torch.float)
        data = data.permute(0, 2, 3, 1).contiguous().view(C * V * M, cropped_length)
        data = data[None, None, :, :]
        data = F.interpolate(
            data, size=(C * V * M, self.window_size),
            mode='bilinear', align_corners=False
        ).squeeze()
        data = data.contiguous().view(C, V, M, self.window_size).permute(0, 3, 1, 2).contiguous().numpy()

        return data

    def _random_rotation_deterministic(self, data_numpy, index, theta=0.3):
        """Deterministic random 2D rotation."""
        C, T, V, M = data_numpy.shape

        # Create sample-specific RNG
        rng = np.random.RandomState(self.augment_seed + index + 100000)

        angle = rng.uniform(-theta, theta)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

        data_numpy = data_numpy.copy()
        for t in range(T):
            for m in range(M):
                xy = data_numpy[:, t, :, m]
                data_numpy[:, t, :, m] = rot @ xy

        return data_numpy

    def _get_bone_data(self, data_numpy):
        """Convert joint data to bone data."""
        bone_pairs = self._get_bone_pairs()

        C, T, V, M = data_numpy.shape
        bone_data = np.zeros_like(data_numpy)

        for v1, v2 in bone_pairs:
            if v1 < V and v2 < V:
                bone_data[:, :, v1, :] = data_numpy[:, :, v1, :] - data_numpy[:, :, v2, :]

        return bone_data

    def _get_bone_pairs(self):
        """Define bone pairs based on selected keypoints."""
        idx_map = {orig: new for new, orig in enumerate(self.selected_keypoints)}

        all_pairs = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (5, 0), (6, 0),
            (7, 5), (9, 7),
            (8, 6), (10, 8),
            (11, 5), (12, 6),
            (13, 11), (15, 13),
            (14, 12), (16, 14),
            (17, 15), (18, 15), (19, 15),
            (20, 16), (21, 16), (22, 16),
        ]

        for i in range(91, 112):
            all_pairs.append((i, 91 if i != 91 else 9))
        for i in range(112, 133):
            all_pairs.append((i, 112 if i != 112 else 10))

        pairs = []
        for v1, v2 in all_pairs:
            if v1 in idx_map and v2 in idx_map:
                pairs.append((idx_map[v1], idx_map[v2]))

        return pairs

    def get_class_weights(self):
        """Calculate class weights for imbalanced dataset."""
        class_counts = np.bincount(self.label, minlength=self.num_class)
        total = len(self.label)
        weights = total / (self.num_class * class_counts + 1e-6)
        return torch.FloatTensor(weights)
