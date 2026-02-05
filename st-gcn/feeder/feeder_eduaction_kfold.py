"""
K-Fold Cross Validation Feeder for EduAction dataset.
Supports stratified K-Fold with deterministic random seeds for reproducibility.
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
    'full_body': list(range(133)),
    'upper_body': list(range(0, 17)) + list(range(23, 133)),
    'body_only': list(range(0, 23)),
    'body_no_feet': list(range(0, 17)),
    'face_hands': list(range(23, 133)),
    'hands_only': list(range(91, 133)),
    'body_hands': list(range(0, 23)) + list(range(91, 133)),
    'upper_body_hands': list(range(0, 11)) + list(range(91, 133)),
    'face_only': list(range(23, 91)),
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
    K-Fold Cross Validation Feeder for EduAction.

    Implements stratified K-Fold splitting with deterministic randomness.
    All random operations use seeded generators for exact reproducibility.

    Args:
        data_dir: Path to EduAction_pose_data folder
        split: 'train' or 'val' (validation fold)
        n_folds: Number of folds (default: 5)
        fold_idx: Current fold index (0 to n_folds-1)
        random_choose: If True, randomly choose temporal segment (deterministic with seed)
        random_move: If True, apply random spatial augmentation (deterministic with seed)
        window_size: Target number of frames
        debug: If True, only use first 100 samples
        seed: Base random seed for fold splitting
        keypoint_subset: Which keypoints to use
        augment_seed: Seed for data augmentation (changes per epoch if needed)
    """

    def __init__(
        self,
        data_dir,
        split='train',
        n_folds=5,
        fold_idx=0,
        random_choose=False,
        random_move=False,
        window_size=-1,
        debug=False,
        seed=42,
        keypoint_subset='full_body',
        augment_seed=None,
        mmap=True
    ):
        self.data_dir = data_dir
        self.split = split
        self.n_folds = n_folds
        self.fold_idx = fold_idx
        self.random_choose = random_choose
        self.random_move = random_move
        self.window_size = window_size
        self.debug = debug
        self.seed = seed
        self.augment_seed = augment_seed if augment_seed is not None else seed

        # Create dedicated random generators for reproducibility
        self.split_rng = random.Random(seed)
        self.augment_rng = np.random.RandomState(self.augment_seed)

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
        self.augment_rng = np.random.RandomState(seed)

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
                T, V, C = data.shape
                data = data.transpose(2, 0, 1)  # (C, T, V)
                data = np.expand_dims(data, axis=-1)  # (C, T, V, 1)

                self.data.append(data)
                self.label.append(sample['label'])
                self.sample_name.append(os.path.basename(sample['path']))

            except Exception as e:
                print(f"Error loading {sample['path']}: {e}")
                continue

        self.max_frames = max(d.shape[1] for d in self.data) if self.data else 0
        print(f"Loaded {len(self.data)} samples. Max frames: {self.max_frames}")

        # Print class distribution
        class_counts = np.bincount(self.label, minlength=self.num_class)
        print(f"Class distribution: {dict(zip(self.classes, class_counts))}")

        if len(self.data) > 0:
            self.C = self.data[0].shape[0]
            self.V = self.data[0].shape[2]
            self.M = self.data[0].shape[3]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_numpy = np.array(self.data[index]).astype(np.float32)
        label = self.label[index]

        C, T, V, M = data_numpy.shape

        if self.window_size > 0:
            if self.random_choose:
                # Deterministic random choose using index-based seed
                data_numpy = self._random_choose_deterministic(data_numpy, index)
            else:
                if T == self.window_size:
                    pass
                elif T < self.window_size:
                    data_numpy = self._auto_pading_deterministic(data_numpy, index)
                else:
                    data_numpy = data_numpy[:, :self.window_size, :, :]

        if self.random_move:
            data_numpy = self._random_move_deterministic(data_numpy, index)

        return data_numpy, label

    def _auto_pading_deterministic(self, data_numpy, index):
        """Deterministic padding using seeded RNG."""
        C, T, V, M = data_numpy.shape
        size = self.window_size

        if T < size:
            # Use index to create reproducible but varied padding
            rng = np.random.RandomState(self.augment_seed + index)
            begin = rng.randint(0, size - T + 1) if self.split == 'train' else 0
            data_numpy_paded = np.zeros((C, size, V, M), dtype=data_numpy.dtype)
            data_numpy_paded[:, begin:begin + T, :, :] = data_numpy
            return data_numpy_paded
        return data_numpy

    def _random_choose_deterministic(self, data_numpy, index):
        """Deterministic random temporal segment selection."""
        C, T, V, M = data_numpy.shape
        size = self.window_size

        if T == size:
            return data_numpy
        elif T < size:
            return self._auto_pading_deterministic(data_numpy, index)
        else:
            rng = np.random.RandomState(self.augment_seed + index)
            begin = rng.randint(0, T - size + 1)
            return data_numpy[:, begin:begin + size, :, :]

    def _random_move_deterministic(self, data_numpy, index):
        """Deterministic random spatial augmentation."""
        C, T, V, M = data_numpy.shape

        # Create seeded RNG for this sample
        rng = np.random.RandomState(self.augment_seed + index + 10000)

        angle_candidate = [-10., -5., 0., 5., 10.]
        scale_candidate = [0.9, 1.0, 1.1]
        transform_candidate = [-0.2, -0.1, 0.0, 0.1, 0.2]

        # Use rng for all random choices
        move_time = 1
        node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
        node = np.append(node, T)
        num_node = len(node)

        A = rng.choice(angle_candidate, num_node)
        S = rng.choice(scale_candidate, num_node)
        T_x = rng.choice(transform_candidate, num_node)
        T_y = rng.choice(transform_candidate, num_node)

        a = np.zeros(T)
        s = np.zeros(T)
        t_x = np.zeros(T)
        t_y = np.zeros(T)

        for i in range(num_node - 1):
            a[node[i]:node[i + 1]] = np.linspace(
                A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
            s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                                  node[i + 1] - node[i])
            t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                                    node[i + 1] - node[i])
            t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                                    node[i + 1] - node[i])

        theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                          [np.sin(a) * s, np.cos(a) * s]])

        data_numpy = data_numpy.copy()  # Don't modify original
        for i_frame in range(T):
            xy = data_numpy[0:2, i_frame, :, :]
            new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
            new_xy[0] += t_x[i_frame]
            new_xy[1] += t_y[i_frame]
            data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

        return data_numpy

    def get_class_weights(self):
        """Calculate class weights for imbalanced dataset."""
        class_counts = np.bincount(self.label, minlength=self.num_class)
        total = len(self.label)
        weights = total / (self.num_class * class_counts + 1e-6)
        return torch.FloatTensor(weights)


def get_kfold_datasets(data_dir, n_folds=5, seed=42, keypoint_subset='full_body',
                       window_size=64, random_choose=True, random_move=True):
    """
    Utility function to get all K-Fold train/val dataset pairs.

    Returns:
        List of (train_dataset, val_dataset) tuples for each fold
    """
    folds = []
    for fold_idx in range(n_folds):
        train_dataset = FeederEduActionKFold(
            data_dir=data_dir,
            split='train',
            n_folds=n_folds,
            fold_idx=fold_idx,
            seed=seed,
            keypoint_subset=keypoint_subset,
            window_size=window_size,
            random_choose=random_choose,
            random_move=random_move,
        )
        val_dataset = FeederEduActionKFold(
            data_dir=data_dir,
            split='val',
            n_folds=n_folds,
            fold_idx=fold_idx,
            seed=seed,
            keypoint_subset=keypoint_subset,
            window_size=window_size,
            random_choose=False,  # No augmentation for validation
            random_move=False,
        )
        folds.append((train_dataset, val_dataset))
    return folds
