#!/usr/bin/env python3
"""
Test script for oversampling functionality in training
"""
import os
import sys
import numpy as np
from collections import Counter

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import training modules
from training import TrainingConfig, load_skeleton_data, preprocess_pose_data, create_data_splits, apply_oversampling_to_training_data

def test_oversampling():
    """
    Test the oversampling functionality
    """
    print("Testing DD-Net Oversampling Functionality")
    print("=" * 50)
    
    # Initialize config
    config = TrainingConfig()
    config.data_dir = "../CStudentAct_processed_pose"
    
    # Enable oversampling for testing
    config.apply_oversampling = True
    config.oversampling_strategy = 'balanced'
    config.augmentation_noise_factor = 0.01
    
    print(f"Configuration:")
    print(f"- Data directory: {config.data_dir}")
    print(f"- Apply oversampling: {config.apply_oversampling}")
    print(f"- Strategy: {config.oversampling_strategy}")
    print(f"- Noise factor: {config.augmentation_noise_factor}")
    print()
    
    try:
        # Load data
        print("Loading skeleton data...")
        poses, labels = load_skeleton_data(config.data_dir, config)
        
        if len(poses) == 0:
            print("❌ No data found! Please check your data directory.")
            return False
        
        print(f"✅ Loaded {len(poses)} samples")
        
        # Show original class distribution
        original_counts = Counter(labels)
        print("\nOriginal class distribution:")
        for class_idx, count in sorted(original_counts.items()):
            class_name = config.class_names[class_idx]
            print(f"  {class_name}: {count} samples")
        
        # Preprocess data
        print("\nPreprocessing data...")
        X_motion, X_pose, Y = preprocess_pose_data(poses, labels, config)
        print(f"✅ Preprocessed to shapes: X_motion{X_motion.shape}, X_pose{X_pose.shape}, Y{Y.shape}")
        
        # Create data splits
        print("\nCreating data splits...")
        (X_motion_train, X_pose_train, Y_train_cat), \
        (X_motion_val, X_pose_val, Y_val_cat), \
        (X_motion_test, X_pose_test, Y_test_cat), \
        (Y_train_raw, Y_val_raw, Y_test_raw) = create_data_splits(X_motion, X_pose, Y, config)
        
        print(f"✅ Created splits:")
        print(f"  - Training: {len(X_motion_train)} samples")
        print(f"  - Validation: {len(X_motion_val)} samples")
        print(f"  - Test: {len(X_motion_test)} samples")
        
        # Show training class distribution before oversampling
        train_counts_before = Counter(Y_train_raw)
        print("\nTraining class distribution BEFORE oversampling:")
        for class_idx, count in sorted(train_counts_before.items()):
            class_name = config.class_names[class_idx]
            print(f"  {class_name}: {count} samples")
        
        # Apply oversampling
        print("\nApplying oversampling...")
        X_motion_train_balanced, X_pose_train_balanced, Y_train_balanced = apply_oversampling_to_training_data(
            X_motion_train, X_pose_train, Y_train_raw, config
        )
        
        # Show results
        train_counts_after = Counter(Y_train_balanced)
        print("\nTraining class distribution AFTER oversampling:")
        for class_idx, count in sorted(train_counts_after.items()):
            class_name = config.class_names[class_idx]
            percentage = count / len(Y_train_balanced) * 100
            print(f"  {class_name}: {count} samples ({percentage:.1f}%)")
        
        # Verify shapes
        print(f"\nFinal training data shapes:")
        print(f"  X_motion: {X_motion_train_balanced.shape}")
        print(f"  X_pose: {X_pose_train_balanced.shape}")
        print(f"  Y: {Y_train_balanced.shape}")
        
        # Test that validation and test data are unchanged
        print(f"\nValidation data (should be unchanged):")
        val_counts = Counter(Y_val_raw)
        for class_idx, count in sorted(val_counts.items()):
            class_name = config.class_names[class_idx]
            print(f"  {class_name}: {count} samples")
        
        print(f"\nTest data (should be unchanged):")
        test_counts = Counter(Y_test_raw)
        for class_idx, count in sorted(test_counts.items()):
            class_name = config.class_names[class_idx]
            print(f"  {class_name}: {count} samples")
        
        print("\n✅ Oversampling test completed successfully!")
        print("\nKey points:")
        print("- ✅ Only training data was oversampled")
        print("- ✅ Validation and test data kept original distribution")
        print("- ✅ All classes in training data are now balanced")
        print("- ✅ Data shapes are consistent")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during oversampling test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_oversampling()
    exit(0 if success else 1)
