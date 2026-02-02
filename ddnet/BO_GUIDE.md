# Bayesian Optimization Guide for DD-Net

## Overview

This guide explains how to use Bayesian Optimization (BO) to automatically find the best hyperparameters for your DD-Net model. The BO will optimize:

- **Learning Rate** (1e-5 to 1e-2)
- **Scheduler Type** (cosine, cosine_restarts, plateau)
- **Number of Filters** (8 to 64)
- **Frame Length** (15 to 60)

## Quick Start

### 1. Run Quick Test (5-10 minutes)
```bash
python quick_bo_test.py
```
This runs a mock optimization to test that everything works.

### 2. Run Real Optimization - Quick Mode (30-60 minutes)
```bash
python bayesian_optimization.py --config quick
```

### 3. Run Real Optimization - Balanced Mode (2-4 hours)
```bash
python bayesian_optimization.py --config balanced
```

### 4. Run Real Optimization - Thorough Mode (6-12 hours)
```bash
python bayesian_optimization.py --config thorough
```

## Configuration Options

| Config | Iterations | CV Folds | Epochs/Trial | Time Estimate | Best For |
|--------|------------|----------|--------------|---------------|----------|
| quick | 10 | 2 | 15 | 30-60 min | Testing, proof of concept |
| balanced | 25 | 3 | 30 | 2-4 hours | Most use cases |
| thorough | 50 | 5 | 40 | 6-12 hours | Final optimization |

## How It Works

1. **Initialization**: BO starts with random hyperparameter combinations
2. **Evaluation**: Each combination is tested using cross-validation
3. **Learning**: BO learns which hyperparameters work better
4. **Optimization**: BO suggests promising hyperparameter combinations
5. **Final Training**: Best hyperparameters are used to train a final model

## Output Files

After optimization, you'll find these files in `bo_results/`:

- `best_hyperparameters.json` - Best hyperparameters found
- `bo_optimization_results.pkl` - Full BO results (can be loaded with skopt)
- `intermediate_results.json` - Results from each iteration
- `optimization_plots.png` - Convergence and evaluation plots
- `best_model_from_bo.h5` - Final model trained with best hyperparameters

## Customizing the Search

### Modify Search Space
Edit `bo_config.py` to change the hyperparameter ranges:

```python
class CustomConfig(BOHyperparameterConfig):
    def __init__(self):
        super().__init__()
        # Customize ranges
        self.lr_min = 1e-4
        self.lr_max = 1e-2
        self.filters_min = 16
        self.filters_max = 32
        # ... other parameters
```

### Add New Hyperparameters
To optimize additional hyperparameters, modify:

1. **Search space** in `bo_config.py`:
```python
def get_search_space(self):
    return [
        # ... existing parameters
        Integer(1, 5, name='new_param'),  # Add new parameter
    ]
```

2. **Objective function** in `bayesian_optimization.py`:
```python
@use_named_args(search_space)
def objective(learning_rate, scheduler, filters, frame_length, new_param):
    # Handle new parameter in config creation
    config.new_param = new_param
    # ... rest of function
```

## Tips for Success

### 1. Start Small
- Use `quick` config first to ensure everything works
- Gradually move to `balanced` and `thorough`

### 2. Monitor Progress
- Check `intermediate_results.json` during optimization
- Look for convergence in the plots

### 3. Resource Management
- BO is memory-intensive (each trial loads data)
- Use `n_jobs=1` unless you have abundant RAM
- Consider smaller `cv_folds` if memory is limited

### 4. Time Management
- Each trial takes ~3-10 minutes depending on data size
- `balanced` config with 25 iterations ≈ 2-4 hours
- Run overnight for thorough optimization

### 5. Result Interpretation
- Focus on cross-validation score (more reliable than single training)
- Check if BO converged (scores stop improving)
- Validate final model performance

## Troubleshooting

### Memory Issues
```python
# In bo_config.py, reduce:
self.cv_folds = 2  # Instead of 3-5
self.bo_batch_size = 4  # Instead of 8
```

### Time Issues
```python
# In bo_config.py, reduce:
self.n_calls = 15  # Instead of 25+
self.bo_epochs = 20  # Instead of 30+
```

### No Improvement
- Increase `n_initial_points` for better exploration
- Expand search ranges if results cluster at boundaries
- Check if your data has enough variation for optimization

## Advanced Usage

### Resume Optimization
```python
from skopt import load
result = load('bo_results/bo_optimization_results.pkl')
# Continue optimization from where it left off
```

### Custom Acquisition Function
```python
# In bo_config.py
self.acquisition_function = 'EI'  # Expected Improvement
# Options: 'gp_hedge', 'EI', 'LCB', 'PI'
```

### Parallel Optimization
```python
# Use multiple cores (requires more memory)
self.n_jobs = -1  # Use all cores
# or
self.n_jobs = 4   # Use 4 cores
```

## Example Workflow

```bash
# 1. Test setup
python quick_bo_test.py

# 2. Quick optimization to get baseline
python bayesian_optimization.py --config quick

# 3. Check results
cat bo_results/best_hyperparameters.json

# 4. Run balanced optimization if quick results look good
python bayesian_optimization.py --config balanced

# 5. Use best model for inference
python -c "
from keras.models import load_model
model = load_model('bo_results/best_model_from_bo.h5')
print('Best model loaded successfully')
"
```

## Expected Results

With good data and sufficient iterations, you should see:

- **5-15% improvement** over default hyperparameters
- **Convergence** in optimization plots
- **Consistent results** across CV folds
- **Learning rate** typically optimizes to 1e-4 to 1e-3 range
- **Frame length** often optimizes to 20-40 range
- **Cosine schedulers** often outperform plateau schedulers

The exact improvements depend on your data quality, model architecture, and optimization settings.
