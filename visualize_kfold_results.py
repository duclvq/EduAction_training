#!/usr/bin/env python
"""
Visualize K-Fold Cross Validation Results for all models.
Compares MGSAN, ST-GCN, and DDNet on EduAction dataset.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Class names
CLASSES = ['drinking', 'lecture', 'play_phone', 'sleeping', 'talking', 'watch_computer', 'writing']
CLASS_ABBREV = ['drink', 'lecture', 'phone', 'sleep', 'talk', 'watch_pc', 'write']

# Result paths
RESULTS = {
    'MGSAN Full Body': 'MGSAN/work_dir/eduaction_kfold/mgsan_joint/kfold_summary.json',
    'MGSAN Upper Body': 'MGSAN/work_dir/eduaction_kfold/mgsan_upper/kfold_summary.json',
    'ST-GCN Full Body': 'st-gcn/work_dir/recognition/eduaction_kfold/ST_GCN/kfold_summary.json',
    'ST-GCN Upper Body': 'st-gcn/work_dir/recognition/eduaction_upper_kfold/ST_GCN/kfold_summary.json',
    'DDNet Full Body': 'MODEL_DDNET/work_dir/ddnet_kfold_full_body/kfold_summary.json',
    'DDNet Upper Body': 'MODEL_DDNET/work_dir/ddnet_kfold_upper_body/kfold_summary.json',
}

def load_results(base_dir):
    """Load all K-Fold results."""
    results = {}
    for name, path in RESULTS.items():
        full_path = os.path.join(base_dir, path)
        if os.path.exists(full_path):
            with open(full_path, 'r') as f:
                results[name] = json.load(f)
            print(f"Loaded: {name}")
        else:
            print(f"Not found: {full_path}")
    return results


def print_summary_table(results):
    """Print summary table of all results."""
    print("\n" + "="*100)
    print("K-FOLD CROSS VALIDATION SUMMARY - ALL MODELS")
    print("="*100)

    # Header
    print(f"\n{'Model':<25} {'Keypoints':<12} {'Mean Acc':<12} {'Std':<10} {'Min':<10} {'Max':<10} {'Time (min)':<12}")
    print("-"*95)

    for name, data in results.items():
        fold_accs = [r['best_val_acc'] for r in data['fold_results']]
        keypoints = data.get('keypoint_subset', data.get('keypoint_config', 'N/A'))
        joint_n = data.get('joint_n', 'N/A')
        if joint_n != 'N/A':
            keypoints = f"{keypoints} ({joint_n})"
        time_min = data['total_training_time_seconds'] / 60

        print(f"{name:<25} {keypoints:<12} {data['mean_accuracy']:>8.2f}%   {data['std_accuracy']:>6.2f}%   "
              f"{min(fold_accs):>6.2f}%   {max(fold_accs):>6.2f}%   {time_min:>8.1f}")

    print("-"*95)

    # Best model
    best_model = max(results.items(), key=lambda x: x[1]['mean_accuracy'])
    print(f"\nBest Model: {best_model[0]} ({best_model[1]['mean_accuracy']:.2f}%)")

    return results


def plot_accuracy_comparison(results, save_path):
    """Plot accuracy comparison bar chart."""
    fig, ax = plt.subplots(figsize=(14, 6))

    # Organize data by body type
    full_body = {k: v for k, v in results.items() if 'Full' in k}
    upper_body = {k: v for k, v in results.items() if 'Upper' in k}

    # Models
    models = ['MGSAN', 'ST-GCN', 'DDNet']
    x = np.arange(len(models))
    width = 0.35

    # Extract data
    full_means = []
    full_stds = []
    upper_means = []
    upper_stds = []

    for model in models:
        full_key = f"{model} Full Body"
        upper_key = f"{model} Upper Body"

        if full_key in results:
            full_means.append(results[full_key]['mean_accuracy'])
            full_stds.append(results[full_key]['std_accuracy'])
        else:
            full_means.append(0)
            full_stds.append(0)

        if upper_key in results:
            upper_means.append(results[upper_key]['mean_accuracy'])
            upper_stds.append(results[upper_key]['std_accuracy'])
        else:
            upper_means.append(0)
            upper_stds.append(0)

    # Plot bars
    bars1 = ax.bar(x - width/2, full_means, width, yerr=full_stds,
                   label='Full Body (133 kp)', capsize=5, color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, upper_means, width, yerr=upper_stds,
                   label='Upper Body (127 kp)', capsize=5, color='#e74c3c', alpha=0.8)

    # Customize
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_title('K-Fold Cross Validation Results Comparison\n(5-Fold, seed=42, window=64 frames)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim(60, 90)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    def add_labels(bars, means, stds):
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.annotate(f'{mean:.1f}%\n(±{std:.1f})',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)

    add_labels(bars1, full_means, full_stds)
    add_labels(bars2, upper_means, upper_stds)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_fold_results(results, save_path):
    """Plot individual fold results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Full Body
    ax1 = axes[0]
    models_full = ['MGSAN Full Body', 'ST-GCN Full Body', 'DDNet Full Body']
    colors = ['#3498db', '#2ecc71', '#9b59b6']

    for i, model in enumerate(models_full):
        if model in results:
            fold_accs = [r['best_val_acc'] for r in results[model]['fold_results']]
            folds = range(1, len(fold_accs) + 1)
            ax1.plot(folds, fold_accs, 'o-', label=model.replace(' Full Body', ''),
                    color=colors[i], markersize=8, linewidth=2)
            ax1.axhline(y=results[model]['mean_accuracy'], color=colors[i],
                       linestyle='--', alpha=0.5)

    ax1.set_xlabel('Fold', fontsize=11)
    ax1.set_ylabel('Accuracy (%)', fontsize=11)
    ax1.set_title('Full Body (133 keypoints)', fontsize=12)
    ax1.set_xticks(range(1, 6))
    ax1.legend(loc='lower right')
    ax1.set_ylim(60, 90)
    ax1.grid(alpha=0.3)

    # Upper Body
    ax2 = axes[1]
    models_upper = ['MGSAN Upper Body', 'ST-GCN Upper Body', 'DDNet Upper Body']

    for i, model in enumerate(models_upper):
        if model in results:
            fold_accs = [r['best_val_acc'] for r in results[model]['fold_results']]
            folds = range(1, len(fold_accs) + 1)
            ax2.plot(folds, fold_accs, 'o-', label=model.replace(' Upper Body', ''),
                    color=colors[i], markersize=8, linewidth=2)
            ax2.axhline(y=results[model]['mean_accuracy'], color=colors[i],
                       linestyle='--', alpha=0.5)

    ax2.set_xlabel('Fold', fontsize=11)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_title('Upper Body (127 keypoints)', fontsize=12)
    ax2.set_xticks(range(1, 6))
    ax2.legend(loc='lower right')
    ax2.set_ylim(60, 90)
    ax2.grid(alpha=0.3)

    plt.suptitle('Per-Fold Accuracy Results', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_per_class_accuracy(results, save_path):
    """Plot per-class accuracy comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Full Body
    ax1 = axes[0]
    models_full = ['MGSAN Full Body', 'ST-GCN Full Body', 'DDNet Full Body']
    model_names = ['MGSAN', 'ST-GCN', 'DDNet']
    x = np.arange(len(CLASSES))
    width = 0.25
    colors = ['#3498db', '#2ecc71', '#9b59b6']

    for i, (model, name) in enumerate(zip(models_full, model_names)):
        if model in results:
            per_class = np.array(results[model]['avg_per_class_accuracy']) * 100
            ax1.bar(x + i*width, per_class, width, label=name, color=colors[i], alpha=0.8)

    ax1.set_ylabel('Accuracy (%)', fontsize=11)
    ax1.set_title('Full Body - Per-Class Accuracy', fontsize=12)
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(CLASS_ABBREV, rotation=45, ha='right')
    ax1.legend(loc='lower right')
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)

    # Upper Body
    ax2 = axes[1]
    models_upper = ['MGSAN Upper Body', 'ST-GCN Upper Body', 'DDNet Upper Body']

    for i, (model, name) in enumerate(zip(models_upper, model_names)):
        if model in results:
            per_class = np.array(results[model]['avg_per_class_accuracy']) * 100
            ax2.bar(x + i*width, per_class, width, label=name, color=colors[i], alpha=0.8)

    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_title('Upper Body - Per-Class Accuracy', fontsize=12)
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(CLASS_ABBREV, rotation=45, ha='right')
    ax2.legend(loc='lower right')
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_confusion_matrices(results, save_path):
    """Plot aggregated confusion matrices."""
    # Dynamically select best performing configuration for each model
    model_groups = {
        'MGSAN': ['MGSAN Full Body', 'MGSAN Upper Body'],
        'ST-GCN': ['ST-GCN Full Body', 'ST-GCN Upper Body'],
        'DDNet': ['DDNet Full Body', 'DDNet Upper Body'],
    }

    selected = []
    for model_name, variants in model_groups.items():
        best_variant = max(
            [v for v in variants if v in results],
            key=lambda x: results[x]['mean_accuracy'],
            default=None
        )
        if best_variant:
            selected.append(best_variant)

    fig, axes = plt.subplots(1, len(selected), figsize=(6*len(selected), 5))
    if len(selected) == 1:
        axes = [axes]

    for ax, model in zip(axes, selected):
        if model in results:
            cm = np.array(results[model]['aggregated_confusion_matrix'])
            # Normalize
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

            sns.heatmap(cm_norm, annot=True, fmt='.0f', cmap='Blues', ax=ax,
                       xticklabels=CLASS_ABBREV, yticklabels=CLASS_ABBREV,
                       cbar_kws={'label': 'Accuracy (%)'})
            ax.set_title(f'{model}\n({results[model]["mean_accuracy"]:.1f}% ± {results[model]["std_accuracy"]:.1f}%)',
                        fontsize=11)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')

    plt.suptitle('Aggregated Confusion Matrices (Normalized %)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_training_time(results, save_path):
    """Plot training time comparison."""
    fig, ax = plt.subplots(figsize=(10, 5))

    models = list(results.keys())
    times = [results[m]['total_training_time_seconds'] / 60 for m in models]
    accs = [results[m]['mean_accuracy'] for m in models]

    # Color by model type
    colors = []
    for m in models:
        if 'MGSAN' in m:
            colors.append('#3498db')
        elif 'ST-GCN' in m:
            colors.append('#2ecc71')
        else:
            colors.append('#9b59b6')

    bars = ax.bar(range(len(models)), times, color=colors, alpha=0.8)

    # Add accuracy labels
    for i, (bar, acc) in enumerate(zip(bars, accs)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Training Time (minutes)', fontsize=11)
    ax.set_title('Training Time Comparison (5-Fold Total)', fontsize=12)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.replace(' Body', '\nBody') for m in models], rotation=0, fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Legend for colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#3498db', label='MGSAN'),
                      Patch(facecolor='#2ecc71', label='ST-GCN'),
                      Patch(facecolor='#9b59b6', label='DDNet')]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def create_summary_report(results, save_path):
    """Create a text summary report."""
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("K-FOLD CROSS VALIDATION SUMMARY REPORT\n")
        f.write("EduAction Dataset - Action Recognition\n")
        f.write("="*80 + "\n\n")

        f.write("EXPERIMENTAL SETTINGS:\n")
        f.write("-" * 40 + "\n")
        f.write("- K-Fold: 5\n")
        f.write("- Random Seed: 42\n")
        f.write("- Window Size: 64 frames\n")
        f.write("- Classes: 7 (drinking, lecture, play_phone, sleeping, talking, watch_computer, writing)\n")
        f.write("- Total Samples: ~350\n\n")

        f.write("RESULTS SUMMARY:\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Model':<25} {'Mean Acc':<12} {'Std':<10} {'Time (min)':<12}\n")
        f.write("-" * 60 + "\n")

        for name, data in results.items():
            time_min = data['total_training_time_seconds'] / 60
            f.write(f"{name:<25} {data['mean_accuracy']:>8.2f}%   {data['std_accuracy']:>6.2f}%   {time_min:>8.1f}\n")

        f.write("\n" + "="*80 + "\n")

        # Best results
        best_full = max([(k, v) for k, v in results.items() if 'Full' in k],
                       key=lambda x: x[1]['mean_accuracy'])
        best_upper = max([(k, v) for k, v in results.items() if 'Upper' in k],
                        key=lambda x: x[1]['mean_accuracy'])
        best_overall = max(results.items(), key=lambda x: x[1]['mean_accuracy'])

        f.write("\nBEST RESULTS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Best Full Body:  {best_full[0]} ({best_full[1]['mean_accuracy']:.2f}%)\n")
        f.write(f"Best Upper Body: {best_upper[0]} ({best_upper[1]['mean_accuracy']:.2f}%)\n")
        f.write(f"Best Overall:    {best_overall[0]} ({best_overall[1]['mean_accuracy']:.2f}%)\n")

        f.write("\n" + "="*80 + "\n")

        # Per-class analysis
        f.write("\nPER-CLASS ACCURACY (Best Model - " + best_overall[0] + "):\n")
        f.write("-" * 40 + "\n")
        per_class = best_overall[1]['avg_per_class_accuracy']
        for cls, acc in zip(CLASSES, per_class):
            f.write(f"  {cls:<20}: {acc*100:>6.1f}%\n")

        # Hardest and easiest classes
        sorted_classes = sorted(zip(CLASSES, per_class), key=lambda x: x[1])
        f.write(f"\nHardest class: {sorted_classes[0][0]} ({sorted_classes[0][1]*100:.1f}%)\n")
        f.write(f"Easiest class: {sorted_classes[-1][0]} ({sorted_classes[-1][1]*100:.1f}%)\n")

    print(f"Saved: {save_path}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'kfold_visualization')
    os.makedirs(output_dir, exist_ok=True)

    print("Loading K-Fold results...")
    results = load_results(base_dir)

    if not results:
        print("No results found!")
        return

    # Print summary
    print_summary_table(results)

    # Generate visualizations
    print("\nGenerating visualizations...")

    plot_accuracy_comparison(results, os.path.join(output_dir, '1_accuracy_comparison.png'))
    plot_fold_results(results, os.path.join(output_dir, '2_fold_results.png'))
    plot_per_class_accuracy(results, os.path.join(output_dir, '3_per_class_accuracy.png'))
    plot_confusion_matrices(results, os.path.join(output_dir, '4_confusion_matrices.png'))
    plot_training_time(results, os.path.join(output_dir, '5_training_time.png'))

    # Create summary report
    create_summary_report(results, os.path.join(output_dir, 'summary_report.txt'))

    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()
