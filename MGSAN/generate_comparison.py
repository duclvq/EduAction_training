"""
Generate comparison visualizations from training results.
Reads CSV files from workdir to produce confusion matrices, accuracy charts, etc.
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support

CLASS_NAMES = ['drinking', 'lecture', 'play_phone', 'sleeping', 'talking', 'watch_computer', 'writing']
SHORT_NAMES = ['drink', 'lect', 'phone', 'sleep', 'talk', 'watch', 'write']

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIGS = {
    'Full Body (133 kp)': {'dir': os.path.join(BASE_DIR, 'workdir/eduaction/mgsan_joint')},
    'Upper Body (127 kp)': {'dir': os.path.join(BASE_DIR, 'workdir/eduaction/mgsan_upper_body')},
    'Body+Hands (65 kp)': {'dir': os.path.join(BASE_DIR, 'workdir/eduaction/mgsan_body_hands')},
}

OUT_DIR = os.path.join(BASE_DIR, 'workdir/eduaction/visualization_results')


def find_best_epoch(work_dir, num_epochs=120):
    """Find the epoch with best test accuracy from CSV files."""
    best_acc = 0
    best_epoch = 1
    for epoch in range(1, num_epochs + 1):
        csv_path = os.path.join(work_dir, f'epoch{epoch}_test_each_class_acc.csv')
        if not os.path.exists(csv_path):
            continue
        with open(csv_path, 'r') as f:
            rows = list(csv.reader(f))
        if len(rows) < 2:
            continue
        cm = np.array([[int(x) for x in r] for r in rows[1:]])
        total = cm.sum()
        acc = np.trace(cm) / total if total > 0 else 0
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
    return best_epoch, best_acc


def load_epoch_result(work_dir, epoch):
    """Load results for a specific epoch."""
    csv_path = os.path.join(work_dir, f'epoch{epoch}_test_each_class_acc.csv')
    with open(csv_path, 'r') as f:
        rows = list(csv.reader(f))
    class_acc = [float(x) for x in rows[0]]
    cm = np.array([[int(x) for x in r] for r in rows[1:]])
    overall = np.trace(cm) / cm.sum()

    # Derive y_true, y_pred from confusion matrix
    y_true, y_pred = [], []
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            y_true.extend([i] * cm[i][j])
            y_pred.extend([j] * cm[i][j])
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, zero_division=0)

    return {
        'cm': cm, 'class_acc': class_acc, 'overall': overall,
        'best_epoch': epoch,
        'precision': prec, 'recall': rec, 'f1': f1,
    }


def get_epoch_accuracies(work_dir, num_epochs=120):
    """Get test accuracy for all epochs."""
    accs = []
    for epoch in range(1, num_epochs + 1):
        csv_path = os.path.join(work_dir, f'epoch{epoch}_test_each_class_acc.csv')
        if not os.path.exists(csv_path):
            accs.append(None)
            continue
        with open(csv_path, 'r') as f:
            rows = list(csv.reader(f))
        if len(rows) < 2:
            accs.append(None)
            continue
        cm = np.array([[int(x) for x in r] for r in rows[1:]])
        total = cm.sum()
        accs.append(np.trace(cm) / total * 100 if total > 0 else 0)
    return accs


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Find best epochs and load results
    results = {}
    for name, cfg in CONFIGS.items():
        best_epoch, best_acc = find_best_epoch(cfg['dir'])
        print(f"{name}: Best Epoch {best_epoch}, Accuracy {best_acc*100:.2f}%")
        results[name] = load_epoch_result(cfg['dir'], best_epoch)

    # ===== 1. Confusion Matrices =====
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for idx, (name, res) in enumerate(results.items()):
        cm_norm = res['cm'].astype('float') / res['cm'].sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.0%', cmap='Blues',
                    xticklabels=SHORT_NAMES, yticklabels=SHORT_NAMES, ax=axes[idx],
                    vmin=0, vmax=1)
        axes[idx].set_title(f"{name}\nAcc: {res['overall']*100:.2f}% (Epoch {res['best_epoch']})", fontsize=11)
        axes[idx].set_xlabel('Predicted')
        if idx == 0:
            axes[idx].set_ylabel('Actual')
    plt.suptitle('Confusion Matrices - Best Epoch', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'confusion_matrices_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print('Saved: confusion_matrices_comparison.png')

    # ===== 2. Per-class Accuracy Comparison =====
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(CLASS_NAMES))
    width = 0.25
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    for idx, (name, res) in enumerate(results.items()):
        offset = (idx - 1) * width
        ax.bar(x + offset, [a * 100 for a in res['class_acc']], width, label=name, color=colors[idx])
    ax.set_xlabel('Class')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Per-Class Accuracy Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 115)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'per_class_accuracy_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print('Saved: per_class_accuracy_comparison.png')

    # ===== 3. Overall Accuracy Bar Chart =====
    fig, ax = plt.subplots(figsize=(8, 5))
    names_list = list(results.keys())
    accs = [results[n]['overall'] * 100 for n in names_list]
    bars = ax.bar(names_list, accs, color=colors)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Overall Test Accuracy Comparison')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'overall_accuracy_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print('Saved: overall_accuracy_comparison.png')

    # ===== 4. F1-Score Comparison =====
    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, (name, res) in enumerate(results.items()):
        offset = (idx - 1) * width
        ax.bar(x + offset, [f * 100 for f in res['f1']], width, label=name, color=colors[idx])
    ax.set_xlabel('Class')
    ax.set_ylabel('F1-Score (%)')
    ax.set_title('Per-Class F1-Score Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 115)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'f1_score_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print('Saved: f1_score_comparison.png')

    # ===== 5. Training Curves =====
    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, (name, cfg) in enumerate(CONFIGS.items()):
        epoch_accs = get_epoch_accuracies(cfg['dir'])
        ax.plot(range(1, 121), epoch_accs, label=name, linewidth=1.5, color=colors[idx])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Test Accuracy Over Training')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(1, 120)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print('Saved: training_curves.png')

    # ===== 6. Detailed Confusion Matrices (individual) =====
    for name, res in results.items():
        fig, axes2 = plt.subplots(1, 2, figsize=(16, 6))

        # Counts
        sns.heatmap(res['cm'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=SHORT_NAMES, yticklabels=SHORT_NAMES, ax=axes2[0])
        axes2[0].set_title(f'{name} - Counts')
        axes2[0].set_xlabel('Predicted')
        axes2[0].set_ylabel('Actual')

        # Normalized
        cm_norm = res['cm'].astype('float') / res['cm'].sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Blues',
                    xticklabels=SHORT_NAMES, yticklabels=SHORT_NAMES, ax=axes2[1])
        axes2[1].set_title(f'{name} - Normalized')
        axes2[1].set_xlabel('Predicted')
        axes2[1].set_ylabel('Actual')

        plt.suptitle(f"Confusion Matrix: {name} (Epoch {res['best_epoch']}, Acc: {res['overall']*100:.2f}%)",
                     fontsize=13, fontweight='bold')
        plt.tight_layout()

        safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', '_')
        plt.savefig(os.path.join(OUT_DIR, f'cm_{safe_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f'Saved: cm_{safe_name}.png')

    # ===== 7. Text Report =====
    report_path = os.path.join(OUT_DIR, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write('=' * 70 + '\n')
        f.write('MGSAN EduAction - Training Results Summary\n')
        f.write('=' * 70 + '\n\n')

        f.write(f'{"Config":<30s} {"Best Epoch":>12s} {"Accuracy":>10s}\n')
        f.write('-' * 55 + '\n')
        for name, res in results.items():
            f.write(f'{name:<30s} {res["best_epoch"]:>12d} {res["overall"]*100:>9.2f}%\n')

        for name, res in results.items():
            f.write('\n' + '=' * 70 + '\n')
            f.write(f'{name} (Best Epoch: {res["best_epoch"]})\n')
            f.write('=' * 70 + '\n')
            f.write(f'Overall Accuracy: {res["overall"]*100:.2f}%\n\n')

            f.write(f'{"Class":<20s} {"Accuracy":>10s} {"Precision":>10s} {"Recall":>10s} {"F1":>10s}\n')
            f.write('-' * 60 + '\n')
            for i, cn in enumerate(CLASS_NAMES):
                f.write(f'{cn:<20s} {res["class_acc"][i]*100:>9.2f}% '
                        f'{res["precision"][i]*100:>9.2f}% '
                        f'{res["recall"][i]*100:>9.2f}% '
                        f'{res["f1"][i]*100:>9.2f}%\n')

            f.write('\nConfusion Matrix:\n')
            header = '          ' + '  '.join([f'{s:>6s}' for s in SHORT_NAMES])
            f.write(header + '\n')
            for i, cn in enumerate(SHORT_NAMES):
                row = f'{cn:>9s} ' + '  '.join([f'{res["cm"][i][j]:>6d}' for j in range(7)])
                f.write(row + '\n')

    print(f'Saved: evaluation_report.txt')

    # Print summary
    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    for name, res in results.items():
        print(f'{name}: {res["overall"]*100:.2f}% (Epoch {res["best_epoch"]})')
    print(f'\nAll files saved to: {OUT_DIR}')


if __name__ == '__main__':
    main()
