"""
Configuration file for Bayesian Optimization
Modify these settings to customize your hyperparameter search
"""

class BOHyperparameterConfig:
    """
    Configuration for Bayesian Optimization hyperparameter search
    """
    def __init__(self):
        # ============ BAYESIAN OPTIMIZATION SETTINGS ============
        
        # Number of BO iterations (more = better results, but slower)
        # Recommended: 30-100 for serious optimization, 10-20 for testing
        self.n_calls = 25
        
        # Initial random evaluations before BO starts
        # Recommended: 5-10 (higher for more dimensions)
        self.n_initial_points = 5
        
        # Random seed for reproducibility
        self.random_state = 42
        
        # Parallel jobs (-1 for all cores, 1 for sequential)
        # Note: Memory usage multiplies with parallel jobs
        self.n_jobs = 1
        
        # ============ CROSS-VALIDATION SETTINGS ============
        
        # Number of CV folds (higher = more robust, but slower)
        # Recommended: 3-5
        self.cv_folds = 3
        
        # Training epochs for each BO trial (keep low for speed)
        # Recommended: 20-40 for BO, then retrain best with more epochs
        self.bo_epochs = 25
        
        # Batch size for BO trials
        self.bo_batch_size = 8
        
        # Early stopping patience for BO trials
        self.early_stopping_patience = 8
        
        # ============ HYPERPARAMETER SEARCH SPACES ============
        
        # Learning Rate Range
        self.lr_min = 1e-5  # Minimum learning rate
        self.lr_max = 1e-2  # Maximum learning rate
        
        # Scheduler Options
        self.scheduler_options = ['cosine', 'cosine_restarts', 'plateau']
        
        # Base Filter Range (will be multiplied in model)
        self.filters_min = 8   # Minimum base filters
        self.filters_max = 64  # Maximum base filters
        
        # Frame Length Range
        self.frame_length_min = 15  # Minimum frames per sequence
        self.frame_length_max = 60  # Maximum frames per sequence
        
        # ============ DATA AND MODEL SETTINGS ============
        
        # Data directory
        self.data_dir = r'D:\py_source\ddnet_classroom\CStudentAct_processed_pose'
        
        # Class names
        self.class_names = ['raising_hand', 'sleeping', 'standing', 'using_phone', 'writing']
        
        # Validation split for train/val during BO
        self.validation_split = 0.2
        
        # ============ OUTPUT SETTINGS ============
        
        # Directory to save BO results
        self.results_dir = 'bo_results'
        
        # File names for outputs
        self.bo_results_file = 'bo_optimization_results.pkl'
        self.best_params_file = 'best_hyperparameters.json'
        self.best_model_file = 'best_model_from_bo.h5'
        self.intermediate_results_file = 'intermediate_results.json'
        
        # ============ FINAL MODEL TRAINING SETTINGS ============
        
        # Epochs for final model training with best params
        self.final_epochs = 100
        
        # Whether to train final model after BO
        self.train_final_model = True
        
        # ============ ADVANCED SETTINGS ============
        
        # Acquisition function for BO
        # Options: 'gp_hedge', 'EI', 'LCB', 'PI'
        self.acquisition_function = 'gp_hedge'
        
        # Number of restart for acquisition optimization
        self.n_restarts_optimizer = 5
        
        # Whether to save intermediate models (uses more disk space)
        self.save_intermediate_models = False

    def get_search_space(self):
        """
        Define the hyperparameter search space
        Returns a list of skopt dimensions
        """
        from skopt.space import Real, Integer, Categorical
        
        return [
            Real(self.lr_min, self.lr_max, name='learning_rate', prior='log-uniform'),
            Categorical(self.scheduler_options, name='scheduler'),
            Integer(self.filters_min, self.filters_max, name='filters'),
            Integer(self.frame_length_min, self.frame_length_max, name='frame_length'),
        ]
    
    def print_config(self):
        """Print current configuration"""
        print("Bayesian Optimization Configuration:")
        print("=" * 50)
        print(f"BO Iterations: {self.n_calls}")
        print(f"Initial Points: {self.n_initial_points}")
        print(f"CV Folds: {self.cv_folds}")
        print(f"BO Epochs: {self.bo_epochs}")
        print(f"Final Epochs: {self.final_epochs}")
        print(f"Parallel Jobs: {self.n_jobs}")
        print()
        print("Search Ranges:")
        print(f"  Learning Rate: {self.lr_min:.1e} - {self.lr_max:.1e}")
        print(f"  Schedulers: {self.scheduler_options}")
        print(f"  Filters: {self.filters_min} - {self.filters_max}")
        print(f"  Frame Length: {self.frame_length_min} - {self.frame_length_max}")
        print("=" * 50)

# ============ PRESET CONFIGURATIONS ============

class QuickTestConfig(BOHyperparameterConfig):
    """Quick configuration for testing (fast but less thorough)"""
    def __init__(self):
        super().__init__()
        self.n_calls = 10
        self.n_initial_points = 3
        self.cv_folds = 2
        self.bo_epochs = 15
        self.final_epochs = 30
        
        # Smaller search space for quick testing
        self.lr_min = 1e-4
        self.lr_max = 1e-2
        self.scheduler_options = ['cosine', 'plateau']
        self.filters_min = 16
        self.filters_max = 32
        self.frame_length_min = 20
        self.frame_length_max = 40

class ThoroughConfig(BOHyperparameterConfig):
    """Thorough configuration for serious optimization (slow but comprehensive)"""
    def __init__(self):
        super().__init__()
        self.n_calls = 20
        self.n_initial_points = 10
        self.cv_folds = 5
        self.bo_epochs = 40
        self.final_epochs = 150
        
        # Full search space
        self.lr_min = 1e-6
        self.lr_max = 1e-1
        self.scheduler_options = ['cosine', 'cosine_restarts', 'plateau']
        self.filters_min = 4
        self.filters_max = 128
        self.frame_length_min = 10
        self.frame_length_max = 80

class BalancedConfig(BOHyperparameterConfig):
    """Balanced configuration (good compromise between speed and thoroughness)"""
    def __init__(self):
        super().__init__()
        self.n_calls = 25
        self.n_initial_points = 5
        self.cv_folds = 3
        self.bo_epochs = 30
        self.final_epochs = 80
        
        # Reasonable search space
        self.lr_min = 1e-5
        self.lr_max = 1e-2
        self.scheduler_options = ['cosine', 'cosine_restarts', 'plateau']
        self.filters_min = 8
        self.filters_max = 64
        self.frame_length_min = 15
        self.frame_length_max = 60

# ============ CONFIGURATION SELECTION ============

def get_config(config_type='balanced'):
    """
    Get configuration based on type
    
    Args:
        config_type: 'quick', 'balanced', 'thorough', or 'custom'
    
    Returns:
        Configuration object
    """
    if config_type == 'quick':
        return QuickTestConfig()
    elif config_type == 'thorough':
        return ThoroughConfig()
    elif config_type == 'balanced':
        return BalancedConfig()
    elif config_type == 'custom':
        return BOHyperparameterConfig()
    else:
        raise ValueError(f"Unknown config type: {config_type}")

if __name__ == "__main__":
    print("Available BO Configurations:")
    print("=" * 60)
    
    configs = ['quick', 'balanced', 'thorough']
    for config_name in configs:
        print(f"\n{config_name.upper()} Configuration:")
        config = get_config(config_name)
        config.print_config()
