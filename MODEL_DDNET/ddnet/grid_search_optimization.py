"""
Simple Grid Search Alternative for Hyperparameter Optimization
This version uses exhaustive grid search instead of Bayesian Optimization
"""
import os
import sys
import json
import itertools
import numpy as np
from datetime import datetime

# Import configuration and training modules
from bo_config import get_config
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

class GridSearchOptimizer:
    """Simple grid search for hyperparameter optimization"""
    
    def __init__(self, config_type='quick'):
        self.config = get_config(config_type)
        self.results = []
        self.cached_data = None
        
        # Create results directory
        os.makedirs(self.config.results_dir, exist_ok=True)
        
        # Define grid search space (smaller than BO for practicality)
        self.param_grid = {
            'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3],
            'scheduler': ['cosine', 'plateau'],
            'filters': [16, 24, 32],
            'frame_length': [20, 30, 40] if config_type == 'quick' else [15, 25, 35, 45]
        }
        
        # Calculate total combinations
        self.total_combinations = np.prod([len(v) for v in self.param_grid.values()])
        print(f"Grid search will test {self.total_combinations} combinations")
    
    def create_model_config(self, params):
        """Create model configuration from parameters"""
        config = TrainingConfig()
        
        # Apply parameters
        config.learning_rate = params['learning_rate']
        config.max_lr = params['learning_rate']
        config.filters = params['filters']
        config.frame_l = params['frame_length']
        config.epochs = self.config.bo_epochs
        config.batch_size = self.config.bo_batch_size
        
        # Set scheduler
        if params['scheduler'] == 'cosine':
            config.use_cosine_scheduler = True
            config.cosine_restarts = False
        else:  # plateau
            config.use_cosine_scheduler = False
            config.cosine_restarts = False
        
        return config
    
    def evaluate_params(self, params, iteration):
        """Evaluate a single parameter combination"""
        print(f"\n{'='*50}")
        print(f"Grid Search: {iteration}/{self.total_combinations}")
        print(f"{'='*50}")
        print(f"Parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        try:
            # Create configuration
            config = self.create_model_config(params)
            
            # Load data (cache on first run)
            if self.cached_data is None:
                print("Loading and preprocessing data...")
                poses, labels = load_skeleton_data(config.data_dir, config)
                if len(poses) == 0:
                    raise ValueError("No data found")
                
                temp_config = TrainingConfig()
                X_motion, X_pose, Y = preprocess_pose_data(poses, labels, temp_config)
                self.cached_data = (X_motion, X_pose, Y)
                print("Data cached")
            else:
                X_motion, X_pose, Y = self.cached_data
            
            # Adjust frame length if needed
            if params['frame_length'] != X_pose.shape[1]:
                print(f"Adjusting frame length to {params['frame_length']}")
                X_pose_adj = []
                X_motion_adj = []
                
                for i in range(len(X_pose)):
                    pose_resized = zoom(X_pose[i], target_l=params['frame_length'],
                                      joints_num=config.joint_n, joints_dim=config.joint_d)
                    motion_feat = get_CG(pose_resized, config)
                    X_pose_adj.append(pose_resized)
                    X_motion_adj.append(motion_feat)
                
                X_pose = np.array(X_pose_adj)
                X_motion = np.array(X_motion_adj)
            
            # Train/validation split
            X_motion_train, X_motion_val, X_pose_train, X_pose_val, Y_train, Y_val = train_test_split(
                X_motion, X_pose, Y, test_size=0.3, stratify=Y, random_state=42
            )
            
            # Convert to categorical
            Y_train_cat = to_categorical(Y_train, num_classes=config.clc_num)
            Y_val_cat = to_categorical(Y_val, num_classes=config.clc_num)
            
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
                    patience=self.config.early_stopping_patience,
                    restore_best_weights=True,
                    verbose=0
                )
            ]
            
            # Add scheduler
            if config.use_cosine_scheduler:
                lr_scheduler = LearningRateScheduler(
                    lambda epoch, lr: cosine_scheduler(
                        epoch, lr, config.epochs, config.min_lr, config.max_lr
                    ), verbose=0
                )
                callbacks.append(lr_scheduler)
            else:
                callbacks.append(
                    ReduceLROnPlateau(
                        monitor='val_loss', factor=0.5, patience=4,
                        min_lr=config.min_lr, verbose=0
                    )
                )
            
            # Train model
            history = model.fit(
                [X_motion_train, X_pose_train], Y_train_cat,
                batch_size=config.batch_size,
                epochs=config.epochs,
                validation_data=([X_motion_val, X_pose_val], Y_val_cat),
                callbacks=callbacks,
                verbose=0
            )
            
            # Get best validation accuracy
            best_val_acc = max(history.history['val_accuracy'])
            
            print(f"Validation Accuracy: {best_val_acc:.4f}")
            
            # Clean up
            del model
            tf.keras.backend.clear_session()
            
            return best_val_acc
            
        except Exception as e:
            print(f"Error: {e}")
            return 0.0
    
    def optimize(self):
        """Run grid search optimization"""
        print("🔍 Starting Grid Search Optimization for DD-Net")
        print("=" * 60)
        self.config.print_config()
        print(f"Testing {self.total_combinations} parameter combinations")
        
        # Generate all combinations
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        
        best_score = 0
        best_params = None
        iteration = 0
        
        # Test each combination
        for combination in itertools.product(*values):
            iteration += 1
            params = dict(zip(keys, combination))
            
            # Evaluate this combination
            score = self.evaluate_params(params, iteration)
            
            # Store result
            result = {
                'iteration': iteration,
                'params': params,
                'accuracy': score,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            self.results.append(result)
            
            # Update best
            if score > best_score:
                best_score = score
                best_params = params
                print(f"🎯 New best score: {best_score:.4f}")
            
            # Save intermediate results
            with open(os.path.join(self.config.results_dir, 'grid_search_results.json'), 'w') as f:
                json.dump({
                    'results': self.results,
                    'best_params': best_params,
                    'best_score': best_score
                }, f, indent=2)
        
        print("\n🎉 Grid Search Completed!")
        print("=" * 50)
        print("Best Parameters:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        print(f"Best Accuracy: {best_score:.4f}")
        
        return best_params, best_score

def main(config_type='quick'):
    """Main function for grid search"""
    print("DD-Net Hyperparameter Optimization with Grid Search")
    print("=" * 60)
    
    # Create optimizer
    optimizer = GridSearchOptimizer(config_type)
    
    # Run optimization
    best_params, best_score = optimizer.optimize()
    
    print(f"\n🏆 Final Results:")
    print(f"Best hyperparameters: {best_params}")
    print(f"Best accuracy: {best_score:.4f}")
    print(f"Results saved in: {optimizer.config.results_dir}/")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='DD-Net Grid Search Optimization')
    parser.add_argument('--config', choices=['quick', 'balanced'], 
                       default='quick', help='Configuration preset')
    
    args = parser.parse_args()
    main(args.config)
