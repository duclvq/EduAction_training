"""
Minimal test of the BO system without full model training
"""
import os
import sys
import numpy as np

# Test basic imports
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    print("✅ scikit-optimize imported")
except ImportError as e:
    print(f"❌ Error importing skopt: {e}")
    sys.exit(1)

try:
    from bo_config import get_config
    print("✅ bo_config imported")
except ImportError as e:
    print(f"❌ Error importing bo_config: {e}")
    sys.exit(1)

# Test configuration
config = get_config('quick')
print(f"✅ Configuration loaded: {config.n_calls} iterations")

# Test search space
search_space = config.get_search_space()
print(f"✅ Search space created with {len(search_space)} dimensions")

# Simple mock objective function
iteration = 0

@use_named_args(search_space)
def mock_objective(learning_rate, scheduler, filters, frame_length):
    global iteration
    iteration += 1
    
    print(f"Iteration {iteration}: LR={learning_rate:.6f}, scheduler={scheduler}, filters={filters}, frames={frame_length}")
    
    # Mock score based on parameters
    score = 0.7 + 0.1 * np.random.random()
    return 1 - score  # Return negative for minimization

def test_minimal_bo():
    """Test minimal BO without model training"""
    print("\n🧪 Testing Minimal Bayesian Optimization")
    print("=" * 50)
    
    # Very quick test
    result = gp_minimize(
        func=mock_objective,
        dimensions=search_space,
        n_calls=5,  # Very few calls
        n_initial_points=2,
        random_state=42,
        verbose=True
    )
    
    print("\n✅ Minimal BO test completed!")
    print(f"Best parameters: {dict(zip([d.name for d in search_space], result.x))}")
    print(f"Best score: {1 - result.fun:.4f}")

if __name__ == "__main__":
    test_minimal_bo()
