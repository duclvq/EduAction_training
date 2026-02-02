"""
Simple Manual Hyperparameter Testing
Test specific hyperparameter combinations without optimization libraries
"""
import os
import sys
import json
import numpy as np
from datetime import datetime

# Import configuration and training modules
from training import check_dependencies

if check_dependencies():
    import tensorflow as tf
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
    from keras.utils import to_categorical
    from sklearn.model_selection import train_test_split
    from ddnet import build_DD_Net
    from training import (
        TrainingConfig, load_skeleton_data, preprocess_pose_data,
        cosine_scheduler, cosine_scheduler_with_restarts
    )
    from utils import zoom, get_CG
else:
    print("❌ Training dependencies not available")
    sys.exit(1)

def test_hyperparameters():
    """Test a few manually selected hyperparameter combinations"""
    
    # Define test configurations
    test_configs = [
        {
            'name': 'baseline',
            'learning_rate': 0.001,
            'scheduler': 'plateau',
            'filters': 16,
            'frame_length': 30,
            'description': 'Default configuration'
        },
        {
            'name': 'cosine_lr',
            'learning_rate': 0.001,
            'scheduler': 'cosine',
            'filters': 16,
            'frame_length': 30,
            'description': 'Cosine scheduler'
        },
        {
            'name': 'higher_lr',
            'learning_rate': 0.005,
            'scheduler': 'cosine',
            'filters': 16,
            'frame_length': 30,
            'description': 'Higher learning rate'
        },
        {
            'name': 'more_filters',
            'learning_rate': 0.001,
            'scheduler': 'cosine',
            'filters': 32,
            'frame_length': 30,
            'description': 'More filters'
        },
        {
            'name': 'shorter_frames',
            'learning_rate': 0.001,
            'scheduler': 'cosine',
            'filters': 16,
            'frame_length': 20,
            'description': 'Shorter sequences'
        }
    ]
    
    print("🧪 Manual Hyperparameter Testing for DD-Net")
    print("=" * 60)
    print(f"Testing {len(test_configs)} configurations")
    
    results = []
    cached_data = None
    
    for i, config_params in enumerate(test_configs, 1):
        print(f"\n{'='*50}")
        print(f"Test {i}/{len(test_configs)}: {config_params['name']}")
        print(f"Description: {config_params['description']}")
        print(f"{'='*50}")
        
        try:
            # Create model configuration
            config = TrainingConfig()
            config.learning_rate = config_params['learning_rate']
            config.max_lr = config_params['learning_rate']
            config.filters = config_params['filters']
            config.frame_l = config_params['frame_length']
            config.epochs = 25  # Reduced for testing
            config.batch_size = 8
            
            # Set scheduler
            if config_params['scheduler'] == 'cosine':
                config.use_cosine_scheduler = True
                config.cosine_restarts = False
            else:  # plateau
                config.use_cosine_scheduler = False
                config.cosine_restarts = False
            
            print(f"Parameters:")
            print(f"  Learning Rate: {config_params['learning_rate']}")
            print(f"  Scheduler: {config_params['scheduler']}")
            print(f"  Filters: {config_params['filters']}")
            print(f"  Frame Length: {config_params['frame_length']}")
            
            # Load data (cache on first run)
            if cached_data is None:
                print("Loading and preprocessing data...")
                poses, labels = load_skeleton_data(config.data_dir, config)
                if len(poses) == 0:
                    raise ValueError("No data found")
                
                temp_config = TrainingConfig()
                X_motion, X_pose, Y = preprocess_pose_data(poses, labels, temp_config)
                cached_data = (X_motion, X_pose, Y)
                print("Data loaded and cached")
            else:
                X_motion, X_pose, Y = cached_data
                print("Using cached data")
            
            # Adjust frame length if needed
            if config_params['frame_length'] != X_pose.shape[1]:
                print(f"Adjusting frame length to {config_params['frame_length']}")
                X_pose_adj = []
                X_motion_adj = []
                
                for j in range(len(X_pose)):
                    pose_resized = zoom(X_pose[j], target_l=config_params['frame_length'],
                                      joints_num=config.joint_n, joints_dim=config.joint_d)
                    motion_feat = get_CG(pose_resized, config)
                    X_pose_adj.append(pose_resized)
                    X_motion_adj.append(motion_feat)
                
                X_pose_test = np.array(X_pose_adj)
                X_motion_test = np.array(X_motion_adj)
            else:
                X_pose_test = X_pose
                X_motion_test = X_motion
            
            # Train/validation split
            X_motion_train, X_motion_val, X_pose_train, X_pose_val, Y_train, Y_val = train_test_split(
                X_motion_test, X_pose_test, Y, test_size=0.3, stratify=Y, random_state=42
            )
            
            # Convert to categorical
            Y_train_cat = to_categorical(Y_train, num_classes=config.clc_num)
            Y_val_cat = to_categorical(Y_val, num_classes=config.clc_num)
            
            print(f"Training samples: {len(X_motion_train)}, Validation: {len(X_motion_val)}")
            
            # Build model
            model = build_DD_Net(config)
            model.compile(
                optimizer=Adam(learning_rate=config.learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Setup callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=8,
                    restore_best_weights=True,
                    verbose=0
                )
            ]
            
            # Add scheduler
            if config.use_cosine_scheduler:
                print("Using cosine scheduler")
                lr_scheduler = LearningRateScheduler(
                    lambda epoch, lr: cosine_scheduler(
                        epoch, lr, config.epochs, config.min_lr, config.max_lr
                    ), verbose=0
                )
                callbacks.append(lr_scheduler)
            else:
                print("Using plateau scheduler")
                callbacks.append(
                    ReduceLROnPlateau(
                        monitor='val_loss', factor=0.5, patience=4,
                        min_lr=config.min_lr, verbose=0
                    )
                )
            
            # Train model
            print("Training model...")
            history = model.fit(
                [X_motion_train, X_pose_train], Y_train_cat,
                batch_size=config.batch_size,
                epochs=config.epochs,
                validation_data=([X_motion_val, X_pose_val], Y_val_cat),
                callbacks=callbacks,
                verbose=1
            )
            
            # Get best validation accuracy
            best_val_acc = max(history.history['val_accuracy'])
            final_val_acc = history.history['val_accuracy'][-1]
            
            print(f"✅ Best Validation Accuracy: {best_val_acc:.4f}")
            print(f"   Final Validation Accuracy: {final_val_acc:.4f}")
            
            # Store result
            result = {
                'name': config_params['name'],
                'description': config_params['description'],
                'params': {k: v for k, v in config_params.items() if k not in ['name', 'description']},
                'best_accuracy': best_val_acc,
                'final_accuracy': final_val_acc,
                'epochs_trained': len(history.history['loss']),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            results.append(result)
            
            # Clean up
            del model
            tf.keras.backend.clear_session()
            
        except Exception as e:
            print(f"❌ Error in configuration {config_params['name']}: {e}")
            result = {
                'name': config_params['name'],
                'description': config_params['description'],
                'params': {k: v for k, v in config_params.items() if k not in ['name', 'description']},
                'best_accuracy': 0.0,
                'final_accuracy': 0.0,
                'epochs_trained': 0,
                'error': str(e),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            results.append(result)
    
    # Summary
    print("\n🎉 Manual Testing Completed!")
    print("=" * 60)
    print("Results Summary:")
    print("-" * 60)
    
    # Sort by best accuracy
    results.sort(key=lambda x: x['best_accuracy'], reverse=True)
    
    for i, result in enumerate(results, 1):
        status = "✅" if result['best_accuracy'] > 0 else "❌"
        print(f"{i}. {status} {result['name']:15} - {result['best_accuracy']:.4f} - {result['description']}")
    
    if results[0]['best_accuracy'] > 0:
        print(f"\n🏆 Best Configuration: {results[0]['name']}")
        print(f"   Accuracy: {results[0]['best_accuracy']:.4f}")
        print(f"   Parameters: {results[0]['params']}")
    
    # Save results
    os.makedirs('manual_test_results', exist_ok=True)
    with open('manual_test_results/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: manual_test_results/results.json")
    
    return results

if __name__ == "__main__":
    test_hyperparameters()
