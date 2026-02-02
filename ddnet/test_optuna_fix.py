#!/usr/bin/env python3
"""
Quick test script for Optuna optimization fix
"""
import os
import sys
import optuna

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_optuna_categorical():
    """Test that Optuna categorical distribution works"""
    print("Testing Optuna Categorical Distribution Fix")
    print("=" * 50)
    
    def objective(trial):
        # Test the exact same pattern used in our optimization
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        scheduler = trial.suggest_categorical('scheduler', ['cosine', 'cosine_restarts', 'plateau'])
        filters = trial.suggest_int('filters', 8, 64)
        frame_length = trial.suggest_int('frame_length', 15, 60)
        
        print(f"Trial {trial.number}:")
        print(f"  LR: {learning_rate:.6f}")
        print(f"  Scheduler: {scheduler}")
        print(f"  Filters: {filters}")
        print(f"  Frame Length: {frame_length}")
        
        # Return a dummy score
        return 0.75 + (filters / 100) + (learning_rate * 10)
    
    try:
        # Create study
        study = optuna.create_study(direction='maximize')
        
        # Run a few trials
        print("Running 3 test trials...")
        study.optimize(objective, n_trials=3)
        
        print("\n✅ Optuna categorical distribution test PASSED!")
        print(f"Best parameters: {study.best_params}")
        print(f"Best value: {study.best_value:.4f}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Optuna test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_optuna_categorical()
    if success:
        print("\n🚀 Ready to run optuna_optimization.py!")
    else:
        print("\n⚠️  Fix needed before running optimization")
    
    exit(0 if success else 1)
