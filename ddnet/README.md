# DD-Net Training

This directory contains training scripts for the DD-Net model on skeleton pose data.

## Files Overview

- `training.py` - Main training script for DD-Net model
- `test_setup.py` - Test script to verify data loading and setup
- `setup.py` - Automated setup script for installing dependencies
- `requirements.txt` - Required Python packages
- `ddnet.py` - DD-Net model architecture
- `utils.py` - Utility functions for data processing

## Quick Start

### 1. Install Dependencies

Option A - Automated setup (recommended):
```bash
python setup.py
```

Option B - Manual installation:
```bash
pip install -r requirements.txt
```

### 2. Verify Setup

Test that everything is working correctly:
```bash
python test_setup.py
```

### 3. Start Training

Run the training script:
```bash
python training.py
```

## Data Format

The training script expects skeleton pose data in the following format:
- Directory: `CStudentAct_processed_pose/`
- Files: `{class_name}_pose.pkl` (e.g., `raising_hand_pose.pkl`)
- Each pickle file contains a dictionary where:
  - Keys: person/sequence IDs
  - Values: numpy arrays of shape `(frames, joints, coordinates)`

## Configuration

The training script uses configuration parameters defined in the `TrainingConfig` class in `training.py`. Key parameters include:

- `frame_l`: Number of frames per sequence (default: 30)
- `joint_n`: Number of joints (default: 48)
- `joint_d`: Joint dimensions (default: 3)
- `batch_size`: Training batch size (default: 16)
- `epochs`: Maximum training epochs (default: 100)
- `learning_rate`: Initial learning rate (default: 0.001)

## Model Architecture

DD-Net (Dual-stream Dynamic Network) consists of:
1. **Motion Stream**: Processes inter-frame motion features
2. **Pose Stream**: Processes raw pose sequences
3. **Feature Fusion**: Combines both streams
4. **Classification Head**: Final classification layers

## Training Features

- **Data Augmentation**: Random frame sampling and temporal alignment
- **Cross-validation**: Automatic train/validation/test splits
- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Adaptive learning rate reduction
- **Model Checkpointing**: Saves best model during training
- **Comprehensive Evaluation**: Classification report and confusion matrix

## Output Files

After training, the following files will be generated:
- `DD_Net_trained.h5` - Best trained model
- `training_history.pkl` - Training history data
- `training_plots.png` - Training accuracy/loss plots
- `confusion_matrix.png` - Confusion matrix visualization

## GPU Support

The training script automatically detects and uses GPU if available. For CPU-only training, the script will work but training will be slower.

## Troubleshooting

### Common Issues

1. **Import Errors**: Run `python setup.py` to install dependencies
2. **Data Loading Errors**: Verify data files exist in `CStudentAct_processed_pose/`
3. **Memory Issues**: Reduce `batch_size` in configuration
4. **GPU Issues**: The script will automatically fall back to CPU

### Data Format Issues

If you have pose data in a different format, you may need to modify the `load_skeleton_data` function in `training.py` to match your data structure.

## Customization

To customize the training for your specific use case:

1. **Modify Class Names**: Update `class_names` in `TrainingConfig`
2. **Adjust Architecture**: Modify `build_DD_Net` function in `ddnet.py`
3. **Change Data Loading**: Update `load_skeleton_data` function
4. **Add New Augmentations**: Extend the data augmentation pipeline

## Performance Tips

1. **Use GPU**: Ensure CUDA-compatible GPU and proper drivers
2. **Batch Size**: Increase batch size if you have sufficient memory
3. **Data Preprocessing**: Consider caching preprocessed data for faster loading
4. **Mixed Precision**: Enable mixed precision training for faster training on modern GPUs

## Citation

If you use this code in your research, please cite the original DD-Net paper and acknowledge this implementation.
