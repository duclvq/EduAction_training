"""
Learning Rate Scheduler Comparison Demo
Demonstrates the difference between cosine scheduler and ReduceLROnPlateau
"""
import numpy as np
import matplotlib.pyplot as plt
import math

def cosine_scheduler(epoch, total_epochs, min_lr=1e-7, max_lr=1e-3):
    """Standard cosine annealing"""
    return min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))

def cosine_scheduler_with_restarts(epoch, total_epochs, min_lr=1e-7, max_lr=1e-3, restart_period=50):
    """Cosine annealing with warm restarts"""
    t_cur = epoch % restart_period
    return min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * t_cur / restart_period))

def plot_learning_rate_schedules():
    """Plot different learning rate schedules for comparison"""
    epochs = 100
    epoch_range = np.arange(0, epochs)
    
    # Parameters
    max_lr = 1e-3
    min_lr = 1e-7
    
    # Cosine annealing
    cosine_lrs = [cosine_scheduler(epoch, epochs, min_lr, max_lr) for epoch in epoch_range]
    
    # Cosine annealing with restarts (period=25)
    cosine_restart_lrs = [cosine_scheduler_with_restarts(epoch, epochs, min_lr, max_lr, 25) 
                         for epoch in epoch_range]
    
    # Simulated ReduceLROnPlateau (drops at certain epochs)
    plateau_lrs = [max_lr] * epochs
    # Simulate plateau reductions at epochs 20, 40, 60, 80
    reduction_epochs = [20, 40, 60, 80]
    current_lr = max_lr
    for i, epoch in enumerate(epoch_range):
        if epoch in reduction_epochs:
            current_lr *= 0.5
        plateau_lrs[i] = current_lr
    
    # Create the plot
    plt.figure(figsize=(15, 10))
    
    # Main comparison plot
    plt.subplot(2, 2, 1)
    plt.plot(epoch_range, cosine_lrs, label='Cosine Annealing', linewidth=2)
    plt.plot(epoch_range, plateau_lrs, label='ReduceLROnPlateau (simulated)', linewidth=2, linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedulers Comparison')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Cosine annealing details
    plt.subplot(2, 2, 2)
    plt.plot(epoch_range, cosine_lrs, label='Standard Cosine', linewidth=2)
    plt.plot(epoch_range, cosine_restart_lrs, label='Cosine with Restarts (period=25)', linewidth=2, linestyle='-.')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Cosine Annealing Variants')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Linear scale for better visualization
    plt.subplot(2, 2, 3)
    plt.plot(epoch_range, cosine_lrs, label='Cosine Annealing', linewidth=2)
    plt.plot(epoch_range, plateau_lrs, label='ReduceLROnPlateau', linewidth=2, linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedulers (Linear Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Zoomed view of first 30 epochs
    plt.subplot(2, 2, 4)
    plt.plot(epoch_range[:30], cosine_lrs[:30], label='Cosine Annealing', linewidth=2, marker='o', markersize=3)
    plt.plot(epoch_range[:30], cosine_restart_lrs[:30], label='Cosine w/ Restarts', linewidth=2, marker='s', markersize=3)
    plt.plot(epoch_range[:30], plateau_lrs[:30], label='ReduceLROnPlateau', linewidth=2, linestyle='--', marker='^', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('First 30 Epochs (Detail View)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('learning_rate_schedulers_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print some statistics
    print("Learning Rate Scheduler Comparison")
    print("=" * 50)
    print(f"Initial LR: {max_lr:.2e}")
    print(f"Final LR (Cosine): {cosine_lrs[-1]:.2e}")
    print(f"Final LR (Plateau): {plateau_lrs[-1]:.2e}")
    print(f"Min LR: {min_lr:.2e}")
    print()
    print("Cosine Annealing Benefits:")
    print("✅ Smooth, gradual decay")
    print("✅ Predictable learning rate schedule")
    print("✅ No need to monitor validation loss")
    print("✅ Can help escape local minima")
    print("✅ Works well with many epochs")
    print()
    print("ReduceLROnPlateau Benefits:")
    print("✅ Adaptive to training progress")
    print("✅ Only reduces when needed")
    print("✅ Can work with variable training times")
    print("✅ Good for fine-tuning")

def print_scheduler_usage():
    """Print usage instructions for the schedulers"""
    print("\n" + "=" * 60)
    print("HOW TO USE COSINE SCHEDULER IN YOUR TRAINING")
    print("=" * 60)
    print()
    print("1. In TrainingConfig, set:")
    print("   config.use_cosine_scheduler = True")
    print("   config.cosine_restarts = False  # or True for restarts")
    print("   config.min_lr = 1e-7")
    print("   config.max_lr = 0.001  # your initial learning rate")
    print()
    print("2. The training script will automatically:")
    print("   - Use cosine annealing instead of ReduceLROnPlateau")
    print("   - Smoothly decrease learning rate over epochs")
    print("   - Print learning rate changes during training")
    print()
    print("3. To switch back to ReduceLROnPlateau:")
    print("   config.use_cosine_scheduler = False")
    print()
    print("4. For cosine annealing with warm restarts:")
    print("   config.use_cosine_scheduler = True")
    print("   config.cosine_restarts = True")
    print()
    print("WHEN TO USE EACH:")
    print("-" * 30)
    print("Use Cosine Scheduler when:")
    print("• You know approximately how many epochs you'll train")
    print("• You want smooth learning rate decay")
    print("• Training for many epochs (50+)")
    print("• Want to avoid manual tuning of patience/factor")
    print()
    print("Use ReduceLROnPlateau when:")
    print("• Training time is uncertain")
    print("• You want adaptive reduction based on performance")
    print("• Fine-tuning pre-trained models")
    print("• Working with small datasets where overfitting is a concern")

if __name__ == "__main__":
    try:
        plot_learning_rate_schedules()
        print_scheduler_usage()
    except ImportError:
        print("Matplotlib not available, skipping plots")
        print_scheduler_usage()
