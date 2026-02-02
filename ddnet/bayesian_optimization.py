"""
Bayesian Optimization for DD-Net Hyperparameter Tuning
Optimizes: learning rate, scheduler type, number of filters, frame length
"""
import os
import sys
import numpy as np
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Check for BO dependencies
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    from skopt import dump, load
    from skopt.plots import plot_convergence, plot_objective, plot_evaluations
    print("✅ scikit-optimize imported successfully")
except ImportError:
    print("❌ scikit-optimize not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-optimize"])
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    from skopt import dump, load
    from skopt.plots import plot_convergence, plot_objective, plot_evaluations

try:
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("⚠️ Matplotlib not available, plots will be skipped")

# Import configuration and training modules
from bo_config import get_config, BOHyperparameterConfig
from training import (
    check_dependencies, TrainingConfig, load_skeleton_data, 
    preprocess_pose_data, create_data_splits, cosine_scheduler, 
    cosine_scheduler_with_restarts
)

if check_dependencies():
    import tensorflow as tf
    from keras.optimizers import Adam
    from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
    from keras.utils import to_categorical
    from ddnet import build_DD_Net, Config
    from utils import zoom, get_CG, sampling_frame
else:
    print("❌ Missing dependencies for training")
    sys.exit(1)

# Global variables for BO
bo_config = None  # Will be set in main()
bo_iteration = 0
bo_results_log = []

def create_model_config(learning_rate, scheduler, filters, frame_length):
    """Create a model configuration based on hyperparameters"""
    config = TrainingConfig()
    
    # Override with BO parameters
    config.learning_rate = learning_rate
    config.max_lr = learning_rate
    config.filters = filters
    config.frame_l = frame_length
    config.epochs = bo_config.bo_epochs
    config.batch_size = bo_config.bo_batch_size
    
    # Set scheduler parameters
    if scheduler == 'cosine':
        config.use_cosine_scheduler = True
        config.cosine_restarts = False
    elif scheduler == 'cosine_restarts':
        config.use_cosine_scheduler = True
        config.cosine_restarts = True
    else:  # plateau
        config.use_cosine_scheduler = False
        config.cosine_restarts = False
    
    # Calculate feature dimensions based on frame length
    config.feat_d = int(config.joint_n * (config.joint_n - 1) / 2)
    
    return config

def cross_validate_model(config, X_motion, X_pose, Y):
    """Perform cross-validation for the given configuration"""
    from sklearn.model_selection import StratifiedKFold
    
    cv_scores = []
    skf = StratifiedKFold(n_splits=bo_config.cv_folds, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_motion, Y)):
        print(f"  Training fold {fold + 1}/{bo_config.cv_folds}")
        
        # Split data
        X_motion_train, X_motion_val = X_motion[train_idx], X_motion[val_idx]
        X_pose_train, X_pose_val = X_pose[train_idx], X_pose[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]
        
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
                patience=bo_config.early_stopping_patience,
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
                    ),
                    verbose=0
                )
            else:
                lr_scheduler = LearningRateScheduler(
                    lambda epoch, lr: cosine_scheduler(
                        epoch, lr, config.epochs, config.min_lr, config.max_lr
                    ),
                    verbose=0
                )
            callbacks.append(lr_scheduler)
        else:
            callbacks.append(
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=4,
                    min_lr=config.min_lr,
                    verbose=0
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
        cv_scores.append(best_val_acc)
        
        # Clean up to save memory
        del model
        tf.keras.backend.clear_session()
    
    return np.mean(cv_scores), np.std(cv_scores)

def objective(**params):
    """
    Objective function for Bayesian Optimization
    Returns negative accuracy (since BO minimizes)
    """
    global bo_iteration, bo_results_log
    bo_iteration += 1
    
    # Extract parameters
    learning_rate = params['learning_rate']
    scheduler = params['scheduler']
    filters = params['filters']
    frame_length = params['frame_length']
    
    print(f"\n{'='*60}")
    print(f"BO Iteration {bo_iteration}/{bo_config.n_calls}")
    print(f"{'='*60}")
    print(f"Testing hyperparameters:")
    print(f"  Learning Rate: {learning_rate:.6f}")
    print(f"  Scheduler: {scheduler}")
    print(f"  Filters: {filters}")
    print(f"  Frame Length: {frame_length}")
    print(f"{'='*60}")
    
    try:
        # Create configuration
        config = create_model_config(learning_rate, scheduler, filters, frame_length)
        
        # Load and preprocess data (cached if possible)
        if not hasattr(objective, 'cached_data'):
            print("Loading and preprocessing data...")
            poses, labels = load_skeleton_data(config.data_dir, config)
            
            if len(poses) == 0:
                print("No data found!")
                return 1.0  # Return high loss
            
            # Use original frame length for preprocessing, then adjust
            temp_config = TrainingConfig()
            X_motion, X_pose, Y = preprocess_pose_data(poses, labels, temp_config)
            objective.cached_data = (X_motion, X_pose, Y)
            print("Data cached for future iterations")
        else:
            X_motion, X_pose, Y = objective.cached_data
            print("Using cached data")
        
        # Adjust frame length if needed
        if frame_length != X_pose.shape[1]:
            print(f"Adjusting frame length from {X_pose.shape[1]} to {frame_length}")
            X_pose_adjusted = []
            X_motion_adjusted = []
            
            for i in range(X_pose.shape[0]):
                # Adjust pose data
                pose = X_pose[i]
                pose_resized = zoom(pose, target_l=frame_length, 
                                  joints_num=config.joint_n, joints_dim=config.joint_d)
                X_pose_adjusted.append(pose_resized)
                
                # Recalculate motion features
                motion_features = get_CG(pose_resized, config)
                X_motion_adjusted.append(motion_features)
            
            X_pose = np.array(X_pose_adjusted)
            X_motion = np.array(X_motion_adjusted)
        
        # Perform cross-validation
        mean_acc, std_acc = cross_validate_model(config, X_motion, X_pose, Y)
        
        # Log results
        result = {
            'iteration': bo_iteration,
            'learning_rate': learning_rate,
            'scheduler': scheduler,
            'filters': filters,
            'frame_length': frame_length,
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        bo_results_log.append(result)
        
        print(f"Results:")
        print(f"  Mean CV Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"  Objective Value: {1 - mean_acc:.4f}")
        
        # Save intermediate results
        save_intermediate_results()
        
        # Return negative accuracy (BO minimizes)
        return 1 - mean_acc
        
    except Exception as e:
        print(f"Error in iteration {bo_iteration}: {e}")
        import traceback
        traceback.print_exc()
        return 1.0  # Return high loss for failed iterations

def save_intermediate_results():
    """Save intermediate results during optimization"""
    results_file = os.path.join(bo_config.results_dir, bo_config.intermediate_results_file)
    with open(results_file, 'w') as f:
        json.dump(bo_results_log, f, indent=2)

def run_bayesian_optimization():
    """Run the Bayesian Optimization"""
    print("🚀 Starting Bayesian Optimization for DD-Net Hyperparameters")
    print("=" * 70)
    print(f"Search Space:")
    print(f"  Learning Rate: {bo_config.lr_min:.1e} to {bo_config.lr_max:.1e} (log-uniform)")
    print(f"  Scheduler: {bo_config.scheduler_options}")
    print(f"  Filters: {bo_config.filters_min} to {bo_config.filters_max}")
    print(f"  Frame Length: {bo_config.frame_length_min} to {bo_config.frame_length_max}")
    print(f"")
    print(f"BO Configuration:")
    print(f"  Total iterations: {bo_config.n_calls}")
    print(f"  Initial random points: {bo_config.n_initial_points}")
    print(f"  CV folds: {bo_config.cv_folds}")
    print(f"  Epochs per trial: {bo_config.bo_epochs}")
    print("=" * 70)
    
    # Clear any cached data
    if hasattr(objective, 'cached_data'):
        delattr(objective, 'cached_data')
    
    # Get search space
    search_space = bo_config.get_search_space()
    
    # Create objective function with proper decorator
    from skopt.utils import use_named_args
    objective_with_args = use_named_args(search_space)(objective)
    
    # Run optimization
    result = gp_minimize(
        func=objective_with_args,
        dimensions=search_space,
        n_calls=bo_config.n_calls,
        n_initial_points=bo_config.n_initial_points,
        random_state=bo_config.random_state,
        n_jobs=bo_config.n_jobs,
        verbose=True
    )
    
    # Save optimization results
    results_path = os.path.join(bo_config.results_dir, bo_config.bo_results_file)
    dump(result, results_path)
    
    print("\n" + "🎉 Bayesian Optimization Completed!" + "\n")
    print("=" * 70)
    print("BEST HYPERPARAMETERS FOUND:")
    print("=" * 70)
    
    best_params = dict(zip([dim.name for dim in search_space], result.x))
    print(f"  Learning Rate: {best_params['learning_rate']:.6f}")
    print(f"  Scheduler: {best_params['scheduler']}")
    print(f"  Filters: {best_params['filters']}")
    print(f"  Frame Length: {best_params['frame_length']}")
    print(f"  Best CV Score: {1 - result.fun:.4f}")
    print(f"  Function Evaluations: {result.func_vals}")
    
    # Save best parameters
    best_params['best_cv_score'] = 1 - result.fun
    best_params['optimization_result'] = str(result)
    
    best_params_file = os.path.join(bo_config.results_dir, bo_config.best_params_file)
    with open(best_params_file, 'w') as f:
        json.dump(best_params, f, indent=2)
    
    # Create plots if available
    if PLOTTING_AVAILABLE:
        create_optimization_plots(result)
    
    return result, best_params

def create_optimization_plots(result):
    """Create optimization plots"""
    print("📊 Creating optimization plots...")
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Convergence plot
        plot_convergence(result, ax=axes[0, 0])
        axes[0, 0].set_title('Convergence Plot')
        
        # Objective function evaluations
        plot_evaluations(result, ax=axes[0, 1])
        axes[0, 1].set_title('Hyperparameter Evaluations')
        
        # Results over iterations
        accuracies = [1 - val for val in result.func_vals]
        axes[1, 0].plot(accuracies, 'b-', marker='o')
        axes[1, 0].set_title('Accuracy Over Iterations')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('CV Accuracy')
        axes[1, 0].grid(True)
        
        # Best accuracy so far
        best_so_far = np.maximum.accumulate(accuracies)
        axes[1, 1].plot(best_so_far, 'g-', marker='s', linewidth=2)
        axes[1, 1].set_title('Best Accuracy So Far')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Best CV Accuracy')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(bo_config.results_dir, 'optimization_plots.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Plots saved to {plot_path}")
        
    except Exception as e:
        print(f"⚠️ Could not create plots: {e}")

def train_final_model_with_best_params(best_params):
    """Train a final model with the best hyperparameters found"""
    print("\n🎯 Training final model with best hyperparameters...")
    print("=" * 60)
    
    # Create config with best parameters
    config = create_model_config(
        best_params['learning_rate'],
        best_params['scheduler'],
        best_params['filters'],
        best_params['frame_length']
    )
    
    # Use more epochs for final training
    config.epochs = bo_config.final_epochs  # Increase for final model
    config.model_save_path = bo_config.best_model_file
    
    # Load data
    poses, labels = load_skeleton_data(config.data_dir, config)
    X_motion, X_pose, Y = preprocess_pose_data(poses, labels, config)
    
    # Create train/test splits
    from training import create_data_splits
    (X_motion_train, X_pose_train, Y_train_cat), \
    (X_motion_val, X_pose_val, Y_val_cat), \
    (X_motion_test, X_pose_test, Y_test_cat), \
    (Y_train_raw, Y_val_raw, Y_test_raw) = create_data_splits(X_motion, X_pose, Y, config)
    
    # Build and train model
    model = build_DD_Net(config)
    model.compile(
        optimizer=Adam(learning_rate=config.learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            config.model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Add scheduler
    if config.use_cosine_scheduler:
        if config.cosine_restarts:
            lr_scheduler = LearningRateScheduler(
                lambda epoch, lr: cosine_scheduler_with_restarts(
                    epoch, lr, config.epochs, config.min_lr, config.max_lr
                ),
                verbose=1
            )
        else:
            lr_scheduler = LearningRateScheduler(
                lambda epoch, lr: cosine_scheduler(
                    epoch, lr, config.epochs, config.min_lr, config.max_lr
                ),
                verbose=1
            )
        callbacks.append(lr_scheduler)
    else:
        callbacks.append(
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=config.min_lr,
                verbose=1
            )
        )
    
    # Train model
    history = model.fit(
        [X_motion_train, X_pose_train], Y_train_cat,
        batch_size=config.batch_size,
        epochs=config.epochs,
        validation_data=([X_motion_val, X_pose_val], Y_val_cat),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate final model
    model.load_weights(config.model_save_path)
    test_loss, test_acc = model.evaluate([X_motion_test, X_pose_test], Y_test_cat, verbose=0)
    
    print(f"\n🎉 Final Model Results:")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Model saved to: {config.model_save_path}")
    
    return model, history, test_acc

def main(config_type='balanced'):
    """
    Main function to run Bayesian Optimization
    
    Args:
        config_type: 'quick', 'balanced', 'thorough', or 'custom'
    """
    global bo_config
    
    print("DD-Net Hyperparameter Optimization with Bayesian Optimization")
    print("=" * 80)
    
    # Load configuration
    bo_config = get_config(config_type)
    
    # Create results directory
    os.makedirs(bo_config.results_dir, exist_ok=True)
    
    # Print configuration
    bo_config.print_config()
    
    # Run Bayesian Optimization
    result, best_params = run_bayesian_optimization()
    
    # Train final model with best parameters (if enabled)
    if bo_config.train_final_model:
        final_model, history, test_acc = train_final_model_with_best_params(best_params)
    else:
        test_acc = "Not computed (final training disabled)"
    
    # Summary
    print("\n" + "🏆 OPTIMIZATION SUMMARY" + "\n")
    print("=" * 80)
    print(f"Configuration used: {config_type}")
    print(f"Best hyperparameters:")
    for param, value in best_params.items():
        if param not in ['best_cv_score', 'optimization_result']:
            print(f"  {param}: {value}")
    print(f"")
    print(f"Cross-validation score: {best_params['best_cv_score']:.4f}")
    print(f"Final test accuracy: {test_acc}")
    print(f"")
    print(f"Results saved in: {bo_config.results_dir}/")
    print("=" * 80)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='DD-Net Bayesian Optimization')
    parser.add_argument('--config', choices=['quick', 'balanced', 'thorough', 'custom'], 
                       default='balanced', help='Configuration preset to use')
    
    args = parser.parse_args()
    main(args.config)
