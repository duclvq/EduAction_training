"""
Alternative Bayesian Optimization using Optuna (more reliable)
This version uses Optuna instead of scikit-optimize
"""
import os
import sys
import json
import numpy as np
import pickle
from datetime import datetime

# Install optuna if not available
try:
    import optuna
    print("✅ Optuna available")
except ImportError:
    print("📦 Installing Optuna...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna"])
    import optuna

# Import your configuration and training modules
from bo_config import get_config

# Check if training dependencies are available  
from training import check_dependencies

if check_dependencies():
    import tensorflow as tf
    from keras.optimizers import Adam
    from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
    from keras.utils import to_categorical
    from ddnet import build_DD_Net
    from training import (
        TrainingConfig, load_skeleton_data, preprocess_pose_data, 
        create_data_splits, cosine_scheduler, cosine_scheduler_with_restarts,
        apply_oversampling_to_training_data
    )
    from utils import zoom, get_CG
else:
    print("❌ Training dependencies not available")
    sys.exit(1)

class OptunaBO:
    """Optuna-based Bayesian Optimization for DD-Net"""
    
    def __init__(self, config_type='balanced'):
        self.config = get_config(config_type)
        self.iteration = 0
        self.cached_data = None
        self.results_log = []
        
        # Create results directory
        os.makedirs(self.config.results_dir, exist_ok=True)
        
        # Create Optuna study with new name to avoid conflicts
        study_name = f'ddnet_hyperopt_v2_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        self.study = optuna.create_study(
            direction='maximize',  # Maximize accuracy
            study_name=study_name,
            storage=f'sqlite:///{self.config.results_dir}/optuna_study_v2.db',
            load_if_exists=True
        )
    
    def create_model_config(self, trial):
        """Create model configuration from Optuna trial"""
        config = TrainingConfig()
        
        # Sample hyperparameters with fixed categorical choices
        learning_rate = trial.suggest_float('learning_rate', self.config.lr_min, self.config.lr_max, log=True)
        scheduler = trial.suggest_categorical('scheduler', ['cosine', 'cosine_restarts', 'plateau'])  # Fixed list
        filters = trial.suggest_int('filters', self.config.filters_min, self.config.filters_max)
        frame_length = trial.suggest_int('frame_length', self.config.frame_length_min, self.config.frame_length_max)
        
        # Apply to config
        config.learning_rate = learning_rate
        config.max_lr = learning_rate
        config.filters = filters
        config.frame_l = frame_length
        config.epochs = self.config.bo_epochs
        config.batch_size = self.config.bo_batch_size
        
        # Set scheduler
        if scheduler == 'cosine':
            config.use_cosine_scheduler = True
            config.cosine_restarts = False
        elif scheduler == 'cosine_restarts':
            config.use_cosine_scheduler = True
            config.cosine_restarts = True
        else:  # plateau
            config.use_cosine_scheduler = False
            config.cosine_restarts = False
        
        return config, learning_rate, scheduler, filters, frame_length
    
    def objective(self, trial):
        """Objective function for Optuna optimization"""
        self.iteration += 1
        
        try:
            # Create configuration
            config, lr, scheduler, filters, frame_length = self.create_model_config(trial)
            
            print(f"\n{'='*60}")
            print(f"Trial {self.iteration} (Optuna)")
            print(f"{'='*60}")
            print(f"Parameters:")
            print(f"  Learning Rate: {lr:.6f}")
            print(f"  Scheduler: {scheduler}")
            print(f"  Filters: {filters}")
            print(f"  Frame Length: {frame_length}")
            
            # Load data (cache on first run)
            if self.cached_data is None:
                print("Loading and preprocessing data...")
                poses, labels = load_skeleton_data(config.data_dir, config)
                if len(poses) == 0:
                    raise ValueError("No data found")
                
                # Use default config for initial preprocessing
                temp_config = TrainingConfig()
                X_motion, X_pose, Y = preprocess_pose_data(poses, labels, temp_config)
                self.cached_data = (X_motion, X_pose, Y)
                print("Data cached for future trials")
            else:
                X_motion, X_pose, Y = self.cached_data
            
            # Adjust data for current frame length if needed
            if frame_length != X_pose.shape[1]:
                print(f"Adjusting frame length to {frame_length}")
                X_pose_adj = []
                X_motion_adj = []
                
                for i in range(len(X_pose)):
                    pose_resized = zoom(X_pose[i], target_l=frame_length, 
                                      joints_num=config.joint_n, joints_dim=config.joint_d)
                    motion_feat = get_CG(pose_resized, config)
                    X_pose_adj.append(pose_resized)
                    X_motion_adj.append(motion_feat)
                
                X_pose = np.array(X_pose_adj)
                X_motion = np.array(X_motion_adj)
            
            # Simple train/validation split for speed
            from sklearn.model_selection import train_test_split
            
            X_motion_train, X_motion_val, X_pose_train, X_pose_val, Y_train, Y_val = train_test_split(
                X_motion, X_pose, Y, test_size=0.3, stratify=Y, random_state=42
            )
            
            # Apply oversampling to training data only
            X_motion_train, X_pose_train, Y_train = apply_oversampling_to_training_data(
                X_motion_train, X_pose_train, Y_train, config
            )
            
            # Convert to categorical
            Y_train_cat = to_categorical(Y_train, num_classes=config.clc_num)
            Y_val_cat = to_categorical(Y_val, num_classes=config.clc_num)
            
            # Build and compile model
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
                if config.cosine_restarts:
                    lr_scheduler = LearningRateScheduler(
                        lambda epoch, lr: cosine_scheduler_with_restarts(
                            epoch, lr, config.epochs, config.min_lr, config.max_lr
                        ), verbose=0
                    )
                else:
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
            
            # Log results
            result = {
                'trial': self.iteration,
                'learning_rate': lr,
                'scheduler': scheduler,
                'filters': filters,
                'frame_length': frame_length,
                'accuracy': best_val_acc,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            self.results_log.append(result)
            
            # Save intermediate results
            with open(os.path.join(self.config.results_dir, 'optuna_results.json'), 'w') as f:
                json.dump(self.results_log, f, indent=2)
            
            print(f"Validation Accuracy: {best_val_acc:.4f}")
            
            # Clean up
            del model
            tf.keras.backend.clear_session()
            
            return best_val_acc
            
        except Exception as e:
            print(f"Error in trial {self.iteration}: {e}")
            return 0.0  # Return low score for failed trials
    
    def optimize(self):
        """Run the optimization"""
        print("🚀 Starting Optuna Bayesian Optimization for DD-Net")
        print("=" * 70)
        self.config.print_config()
        
        # Run optimization
        self.study.optimize(self.objective, n_trials=self.config.n_calls)
        
        # Get best results
        best_trial = self.study.best_trial
        best_params = best_trial.params
        best_score = best_trial.value
        
        print("\n🎉 Optimization Completed!")
        print("=" * 50)
        print("Best Parameters:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        print(f"Best Accuracy: {best_score:.4f}")
        
        # Save results
        results = {
            'best_params': best_params,
            'best_score': best_score,
            'all_trials': [
                {'params': trial.params, 'value': trial.value} 
                for trial in self.study.trials
            ]
        }
        
        with open(os.path.join(self.config.results_dir, 'final_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        return best_params, best_score

def main(config_type='balanced'):
    """Main function for Optuna optimization"""
    print("DD-Net Hyperparameter Optimization with Optuna")
    print("=" * 60)
    
    # Create optimizer
    optimizer = OptunaBO(config_type)
    
    # Run optimization
    best_params, best_score = optimizer.optimize()
    
    print(f"\n🏆 Final Results:")
    print(f"Best hyperparameters: {best_params}")
    print(f"Best accuracy: {best_score:.4f}")
    print(f"Results saved in: {optimizer.config.results_dir}/")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='DD-Net Optuna Optimization')
    parser.add_argument('--config', choices=['quick', 'balanced', 'thorough'], 
                       default='balanced', help='Configuration preset')
    
    args = parser.parse_args()
    main(args.config)
