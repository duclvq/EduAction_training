"""
Comprehensive visualization for all EduAction training results.
Parses log files from MGSAN and ST-GCN, generates training curves,
accuracy comparisons, and per-class analysis.
"""
import os
import re
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150

CLASS_NAMES = ['drinking', 'lecture', 'play_phone', 'sleeping', 'talking', 'watch_computer', 'writing']

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'visualization_results')


def parse_log_top1(log_path):
    """Parse log.txt and extract per-epoch Top1 accuracy.
    Handles multiple training runs in the same log by detecting epoch resets.
    Returns list of runs, each run is a list of (epoch, accuracy) tuples.
    """
    runs = []
    current_run = []
    current_epoch = None
    last_epoch = -1

    with open(log_path, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        # Match epoch line: "Eval epoch: N" or "Epoch: N"
        epoch_match = re.search(r'Eval epoch:\s*(\d+)', line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            continue

        # Match Top1 line
        top1_match = re.search(r'Top1:\s*([\d.]+)%', line)
        if top1_match:
            acc = float(top1_match.group(1))

            # For MGSAN logs that don't have "Eval epoch" lines,
            # infer epoch from sequential order
            if current_epoch is None:
                # Count how many Top1 entries we've seen so far in this run
                epoch_num = len(current_run) + 1
            else:
                epoch_num = current_epoch

            # Detect new run: epoch resets
            if current_run and epoch_num <= last_epoch - 5:
                runs.append(current_run)
                current_run = []

            current_run.append((epoch_num, acc))
            last_epoch = epoch_num
            current_epoch = None

    if current_run:
        runs.append(current_run)

    return runs


def parse_mgsan_log(log_path):
    """Parse MGSAN log that has epoch in training lines but not eval lines.
    Format:
      [ date ] Eval Epoch: N, ...
      [ date ] Top1: XX.XX%
    or:
      [ date ] Training epoch: N
      ...
      [ date ] Top1: XX.XX%
    """
    runs = []
    current_run = []
    last_epoch = -1

    with open(log_path, 'r') as f:
        lines = f.readlines()

    current_epoch = None
    epoch_counter = 0

    for line in lines:
        # Try to match "Eval Epoch" or "Eval epoch"
        epoch_match = re.search(r'(?:Eval|Training)\s+[Ee]poch:\s*(\d+)', line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            continue

        # Match Top1
        top1_match = re.search(r'Top1:\s*([\d.]+)%', line)
        if top1_match:
            acc = float(top1_match.group(1))

            if current_epoch is not None:
                epoch_num = current_epoch
            else:
                epoch_counter += 1
                epoch_num = epoch_counter

            # Detect new run
            if current_run and epoch_num < last_epoch - 5:
                runs.append(current_run)
                current_run = []
                epoch_counter = epoch_num

            current_run.append((epoch_num, acc))
            last_epoch = epoch_num
            current_epoch = None

    if current_run:
        runs.append(current_run)

    return runs


def parse_mgsan_csv(csv_dir, num_classes=7):
    """Parse MGSAN per-class accuracy CSV files.
    Returns best epoch's confusion matrix and per-class accuracies.
    CSV format: row 0 = per-class acc, rows 1-7 = confusion matrix.
    """
    csv_files = sorted([f for f in os.listdir(csv_dir) if f.endswith('_each_class_acc.csv')])
    if not csv_files:
        return None, None, None

    best_acc = 0
    best_cm = None
    best_class_acc = None
    best_epoch = None

    for csv_file in csv_files:
        epoch_match = re.search(r'epoch(\d+)', csv_file)
        epoch = int(epoch_match.group(1)) if epoch_match else 0

        filepath = os.path.join(csv_dir, csv_file)
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)

        if len(rows) < 2:
            continue

        # Row 0: per-class accuracies
        class_acc = [float(x) for x in rows[0]]
        overall_acc = np.mean(class_acc)

        # Rows 1-7: confusion matrix
        cm = []
        for row in rows[1:num_classes + 1]:
            cm.append([int(float(x)) for x in row])
        cm = np.array(cm)

        if overall_acc > best_acc:
            best_acc = overall_acc
            best_cm = cm
            best_class_acc = class_acc
            best_epoch = epoch

    return best_cm, best_class_acc, best_epoch


def get_all_results():
    """Collect all training results from MGSAN and ST-GCN."""
    results = {}

    # MGSAN configs
    mgsan_configs = {
        'MGSAN Full Body (133kp)': 'MGSAN/workdir/eduaction/mgsan_joint',
        'MGSAN Upper Body (127kp)': 'MGSAN/workdir/eduaction/mgsan_upper_body',
        'MGSAN Body+Hands (65kp)': 'MGSAN/workdir/eduaction/mgsan_body_hands',
    }

    for name, rel_path in mgsan_configs.items():
        work_dir = os.path.join(BASE_DIR, rel_path)
        log_path = os.path.join(work_dir, 'log.txt')
        if not os.path.exists(log_path):
            continue

        runs = parse_mgsan_log(log_path)
        # Use the last complete run (most recent training)
        if runs:
            last_run = runs[-1]
            best_epoch, best_acc = max(last_run, key=lambda x: x[1])

            # Try to get CSV data
            cm, class_acc, csv_epoch = parse_mgsan_csv(work_dir)

            results[name] = {
                'model': 'MGSAN',
                'runs': runs,
                'best_acc': best_acc,
                'best_epoch': best_epoch,
                'confusion_matrix': cm,
                'class_acc': class_acc,
                'work_dir': work_dir,
            }

    # ST-GCN configs
    stgcn_configs = {
        'ST-GCN Full Body (133kp)': 'st-gcn/work_dir/recognition/eduaction/ST_GCN',
        'ST-GCN Upper Body (127kp)': 'st-gcn/work_dir/recognition/eduaction_upper/ST_GCN',
    }

    for name, rel_path in stgcn_configs.items():
        work_dir = os.path.join(BASE_DIR, rel_path)
        log_path = os.path.join(work_dir, 'log.txt')
        if not os.path.exists(log_path):
            continue

        runs = parse_log_top1(log_path)
        if runs:
            # Find best across all runs
            all_results = []
            for run_idx, run in enumerate(runs):
                for epoch, acc in run:
                    all_results.append((run_idx, epoch, acc))

            best_run_idx, best_epoch, best_acc = max(all_results, key=lambda x: x[2])

            results[name] = {
                'model': 'ST-GCN',
                'runs': runs,
                'best_acc': best_acc,
                'best_epoch': best_epoch,
                'best_run_idx': best_run_idx,
                'confusion_matrix': None,
                'class_acc': None,
                'work_dir': work_dir,
            }

    return results


def plot_training_curves(results, save_path):
    """Plot training curves for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    colors_mgsan = ['#e74c3c', '#3498db', '#2ecc71']
    colors_stgcn = ['#9b59b6', '#f39c12']

    mgsan_idx = 0
    stgcn_idx = 0

    # Plot all models on left, grouped by model type on right
    for name, data in results.items():
        # Use the last run for the training curve
        last_run = data['runs'][-1]
        epochs = [e for e, a in last_run]
        accs = [a for e, a in last_run]

        if data['model'] == 'MGSAN':
            color = colors_mgsan[mgsan_idx % len(colors_mgsan)]
            mgsan_idx += 1
            linestyle = '-'
        else:
            color = colors_stgcn[stgcn_idx % len(colors_stgcn)]
            stgcn_idx += 1
            linestyle = '--'

        label = f"{name} (best: {data['best_acc']:.2f}%)"

        # Left: all curves
        axes[0].plot(epochs, accs, color=color, linestyle=linestyle, label=label, alpha=0.8, linewidth=1.5)

        # Right: smoothed curves (moving average)
        window = min(5, len(accs))
        if window > 1:
            smoothed = np.convolve(accs, np.ones(window)/window, mode='valid')
            smooth_epochs = epochs[window-1:]
        else:
            smoothed = accs
            smooth_epochs = epochs
        axes[1].plot(smooth_epochs, smoothed, color=color, linestyle=linestyle,
                     label=label, alpha=0.9, linewidth=2)

    for ax in axes:
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Top-1 Accuracy (%)')
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)

    axes[0].set_title('Training Curves (Raw)')
    axes[1].set_title('Training Curves (Smoothed, window=5)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")


def plot_best_accuracy_comparison(results, save_path):
    """Bar chart comparing best accuracy across all models."""
    names = list(results.keys())
    accs = [results[n]['best_acc'] for n in names]

    # Sort by accuracy
    sorted_pairs = sorted(zip(names, accs), key=lambda x: x[1], reverse=True)
    names = [p[0] for p in sorted_pairs]
    accs = [p[1] for p in sorted_pairs]

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = []
    for name in names:
        if 'MGSAN' in name:
            colors.append('#3498db')
        else:
            colors.append('#e74c3c')

    bars = ax.barh(range(len(names)), accs, color=colors, edgecolor='white', linewidth=0.5)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel('Best Top-1 Accuracy (%)')
    ax.set_title('Best Accuracy Comparison - EduAction Dataset')
    ax.set_xlim(0, 100)

    # Add value labels
    for bar, acc in zip(bars, accs):
        ax.text(acc + 0.5, bar.get_y() + bar.get_height()/2,
                f'{acc:.2f}%', va='center', fontsize=10, fontweight='bold')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='MGSAN'),
        Patch(facecolor='#e74c3c', label='ST-GCN'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Best accuracy comparison saved to {save_path}")


def plot_confusion_matrices(results, save_path):
    """Plot confusion matrices for models that have them (from CSV)."""
    cm_results = {k: v for k, v in results.items() if v['confusion_matrix'] is not None}

    if not cm_results:
        print("No confusion matrices available (no CSV files found)")
        return

    n_plots = len(cm_results)
    fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 7))
    if n_plots == 1:
        axes = [axes]

    import seaborn as sns

    for idx, (name, data) in enumerate(cm_results.items()):
        cm = data['confusion_matrix']
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    ax=axes[idx])
        axes[idx].set_title(f'{name}\n(Best: {data["best_acc"]:.2f}%)')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].tick_params(axis='y', rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrices saved to {save_path}")


def plot_per_class_accuracy(results, save_path):
    """Plot per-class accuracy comparison for models with CSV data."""
    acc_results = {k: v for k, v in results.items() if v['class_acc'] is not None}

    if not acc_results:
        print("No per-class accuracy data available")
        return

    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(CLASS_NAMES))
    n_models = len(acc_results)
    width = 0.8 / n_models

    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12']

    for idx, (name, data) in enumerate(acc_results.items()):
        class_acc = [a * 100 for a in data['class_acc']]  # Convert to percentage
        offset = (idx - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, class_acc, width, label=name, color=colors[idx % len(colors)])

        for bar, acc in zip(bars, class_acc):
            if acc > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{acc:.0f}', ha='center', va='bottom', fontsize=7)

    ax.set_xlabel('Class')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Per-Class Accuracy Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Per-class accuracy saved to {save_path}")


def plot_stgcn_all_runs(results, save_path):
    """Plot ST-GCN training curves showing all runs."""
    stgcn_results = {k: v for k, v in results.items() if v['model'] == 'ST-GCN'}

    if not stgcn_results:
        print("No ST-GCN results found")
        return

    n_configs = len(stgcn_results)
    fig, axes = plt.subplots(1, n_configs, figsize=(8 * n_configs, 6))
    if n_configs == 1:
        axes = [axes]

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']

    for ax_idx, (name, data) in enumerate(stgcn_results.items()):
        for run_idx, run in enumerate(data['runs']):
            epochs = [e for e, a in run]
            accs = [a for e, a in run]
            best_e, best_a = max(run, key=lambda x: x[1])

            color = colors[run_idx % len(colors)]
            label = f'Run {run_idx + 1} (best: {best_a:.2f}% @ ep{best_e})'
            axes[ax_idx].plot(epochs, accs, color=color, label=label, alpha=0.8, linewidth=1.5)
            axes[ax_idx].axhline(y=best_a, color=color, linestyle=':', alpha=0.4)

        axes[ax_idx].set_title(name)
        axes[ax_idx].set_xlabel('Epoch')
        axes[ax_idx].set_ylabel('Top-1 Accuracy (%)')
        axes[ax_idx].legend(fontsize=9)
        axes[ax_idx].grid(True, alpha=0.3)
        axes[ax_idx].set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ST-GCN all runs saved to {save_path}")


def plot_mgsan_all_runs(results, save_path):
    """Plot MGSAN training curves showing all runs."""
    mgsan_results = {k: v for k, v in results.items() if v['model'] == 'MGSAN'}

    if not mgsan_results:
        print("No MGSAN results found")
        return

    n_configs = len(mgsan_results)
    fig, axes = plt.subplots(1, n_configs, figsize=(8 * n_configs, 6))
    if n_configs == 1:
        axes = [axes]

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']

    for ax_idx, (name, data) in enumerate(mgsan_results.items()):
        for run_idx, run in enumerate(data['runs']):
            epochs = [e for e, a in run]
            accs = [a for e, a in run]
            best_e, best_a = max(run, key=lambda x: x[1])

            color = colors[run_idx % len(colors)]
            label = f'Run {run_idx + 1} (best: {best_a:.2f}% @ ep{best_e})'
            axes[ax_idx].plot(epochs, accs, color=color, label=label, alpha=0.8, linewidth=1.5)
            axes[ax_idx].axhline(y=best_a, color=color, linestyle=':', alpha=0.4)

        axes[ax_idx].set_title(name)
        axes[ax_idx].set_xlabel('Epoch')
        axes[ax_idx].set_ylabel('Top-1 Accuracy (%)')
        axes[ax_idx].legend(fontsize=9)
        axes[ax_idx].grid(True, alpha=0.3)
        axes[ax_idx].set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"MGSAN all runs saved to {save_path}")


def generate_summary_report(results, save_path):
    """Generate a text summary report."""
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("EduAction Training Results Summary\n")
        f.write("=" * 70 + "\n\n")

        f.write("Dataset: EduAction (7 classes, 350 samples)\n")
        f.write("Split: 70% train (245) / 30% test (105), seed=42, stratified\n")
        f.write(f"Classes: {', '.join(CLASS_NAMES)}\n\n")

        # Summary table
        f.write("-" * 70 + "\n")
        f.write(f"{'Model':<35} {'Best Acc':>10} {'Best Epoch':>12} {'Runs':>6}\n")
        f.write("-" * 70 + "\n")

        # Sort by best accuracy
        sorted_results = sorted(results.items(), key=lambda x: x[1]['best_acc'], reverse=True)

        for name, data in sorted_results:
            n_runs = len(data['runs'])
            f.write(f"{name:<35} {data['best_acc']:>9.2f}% {data['best_epoch']:>10d} {n_runs:>6d}\n")

        f.write("-" * 70 + "\n\n")

        # Detailed per-model info
        for name, data in sorted_results:
            f.write(f"\n{'='*50}\n")
            f.write(f"{name}\n")
            f.write(f"{'='*50}\n")
            f.write(f"  Best Accuracy: {data['best_acc']:.2f}% (epoch {data['best_epoch']})\n")
            f.write(f"  Number of training runs: {len(data['runs'])}\n")

            for run_idx, run in enumerate(data['runs']):
                best_e, best_a = max(run, key=lambda x: x[1])
                f.write(f"  Run {run_idx + 1}: {len(run)} epochs, best {best_a:.2f}% @ epoch {best_e}\n")

            if data['class_acc'] is not None:
                f.write(f"\n  Per-Class Accuracy:\n")
                for cls, acc in zip(CLASS_NAMES, data['class_acc']):
                    f.write(f"    {cls:20s}: {acc*100:.2f}%\n")

            if data['confusion_matrix'] is not None:
                f.write(f"\n  Confusion Matrix:\n")
                f.write(f"    {'':20s} " + " ".join(f"{c:>6s}" for c in [cn[:6] for cn in CLASS_NAMES]) + "\n")
                for i, row in enumerate(data['confusion_matrix']):
                    f.write(f"    {CLASS_NAMES[i]:20s} " + " ".join(f"{v:>6d}" for v in row) + "\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("End of Report\n")
        f.write("=" * 70 + "\n")

    print(f"Summary report saved to {save_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Collecting results from all models...")
    results = get_all_results()

    if not results:
        print("No results found! Check that work directories exist.")
        return

    print(f"\nFound {len(results)} model configurations:")
    for name, data in sorted(results.items(), key=lambda x: x[1]['best_acc'], reverse=True):
        print(f"  {name}: best {data['best_acc']:.2f}% (epoch {data['best_epoch']}, {len(data['runs'])} runs)")

    print("\nGenerating visualizations...")

    # 1. Training curves (all models)
    plot_training_curves(results, os.path.join(OUTPUT_DIR, 'training_curves_all.png'))

    # 2. Best accuracy comparison
    plot_best_accuracy_comparison(results, os.path.join(OUTPUT_DIR, 'best_accuracy_comparison.png'))

    # 3. ST-GCN individual runs
    plot_stgcn_all_runs(results, os.path.join(OUTPUT_DIR, 'stgcn_training_runs.png'))

    # 4. MGSAN individual runs
    plot_mgsan_all_runs(results, os.path.join(OUTPUT_DIR, 'mgsan_training_runs.png'))

    # 5. Confusion matrices (for models with CSV data)
    plot_confusion_matrices(results, os.path.join(OUTPUT_DIR, 'confusion_matrices.png'))

    # 6. Per-class accuracy comparison
    plot_per_class_accuracy(results, os.path.join(OUTPUT_DIR, 'per_class_accuracy.png'))

    # 7. Summary report
    generate_summary_report(results, os.path.join(OUTPUT_DIR, 'summary_report.txt'))

    print(f"\nAll visualizations saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
