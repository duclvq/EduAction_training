# Oversampling in DD-Net Training

## Overview
This document explains how oversampling is implemented in the DD-Net training pipeline to handle class imbalance while maintaining proper evaluation practices.

## Problem: Class Imbalance
Original class distribution in CStudentAct dataset:
- **raising_hand**: 39 persons
- **sleeping**: 33 persons  
- **standing**: 46 persons
- **using_phone**: **18 persons** (smallest class - 58% fewer samples than largest)
- **writing**: 53 persons

## Solution: Smart Oversampling Strategy

### Key Principles
1. **Training Only**: Oversampling is applied ONLY to training data
2. **Test Integrity**: Validation and test sets maintain original distribution
3. **Realistic Evaluation**: Models are evaluated on real-world class distribution

### Implementation Details

#### 1. Configuration Options
```python
# In TrainingConfig class
self.apply_oversampling = True              # Enable/disable oversampling
self.oversampling_strategy = 'balanced'     # 'balanced' or 'target_count'
self.target_samples_per_class = None        # Specific target (None for balanced)
self.augmentation_noise_factor = 0.01       # 1% noise for augmented samples
```

#### 2. Oversampling Process
1. **Split First**: Create train/validation/test splits with original data
2. **Analyze Distribution**: Count samples per class in training set
3. **Determine Target**: Use maximum class count as target for all classes
4. **Generate Samples**: Create additional samples through augmentation
5. **Add Noise**: Apply small random noise (1%) to prevent overfitting
6. **Shuffle**: Randomize the final training set

#### 3. Augmentation Method
- **Source Selection**: Randomly choose existing samples to duplicate
- **Noise Addition**: Add Gaussian noise (σ = 1% of data range)
- **Preserve Structure**: Maintain temporal and spatial relationships

### Usage Example

```python
# In training script
config = TrainingConfig()
config.apply_oversampling = True

# Normal training pipeline
poses, labels = load_skeleton_data(config.data_dir, config)
X_motion, X_pose, Y = preprocess_pose_data(poses, labels, config)

# Create splits
(X_motion_train, X_pose_train, Y_train_cat), \
(X_motion_val, X_pose_val, Y_val_cat), \
(X_motion_test, X_pose_test, Y_test_cat), \
(Y_train_raw, Y_val_raw, Y_test_raw) = create_data_splits(X_motion, X_pose, Y, config)

# Apply oversampling ONLY to training data
X_motion_train, X_pose_train, Y_train_raw = apply_oversampling_to_training_data(
    X_motion_train, X_pose_train, Y_train_raw, config
)
```

### Expected Results

#### Before Oversampling (Training Set)
```
raising_hand: 31 samples (26.3%)
sleeping:     26 samples (22.0%)
standing:     37 samples (31.4%)
using_phone:  14 samples (11.9%)  ← Severely underrepresented
writing:      42 samples (35.6%)
Total: 150 samples
```

#### After Oversampling (Training Set)
```
raising_hand: 42 samples (20.0%)
sleeping:     42 samples (20.0%)
standing:     42 samples (20.0%)
using_phone:  42 samples (20.0%)  ← Now balanced
writing:      42 samples (20.0%)
Total: 210 samples
```

#### Validation/Test Sets (Unchanged)
```
Original distribution maintained for realistic evaluation
```

### Benefits

1. **Improved Performance**: Better recognition of minority classes
2. **Balanced Learning**: Each class gets equal training attention
3. **Realistic Evaluation**: Test metrics reflect real-world performance
4. **Reduced Overfitting**: Small noise prevents exact duplicates

### Testing

Run the test script to verify functionality:
```bash
cd ddnet
python test_oversampling.py
```

### Integration with Hyperparameter Optimization

All optimization scripts (manual, Optuna, grid search) now support oversampling:
- Oversampling is applied consistently across all hyperparameter configurations
- Each trial gets balanced training data
- Evaluation remains on original test distribution

This ensures fair comparison between different hyperparameter combinations while maintaining evaluation integrity.
