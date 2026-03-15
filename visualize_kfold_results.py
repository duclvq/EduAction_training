#!/usr/bin/env python
"""
Visualize K-Fold Cross Validation Results for all models.
Compares MGSAN, ST-GCN, DDNet across 4 keypoint variants:
  - Full Body   (133 kp, MediaPipe)
  - Upper Body  (127 kp, MediaPipe)
  - Body-17     ( 17 kp, MediaPipe)
  - COCO-17     ( 17 kp, ViTPose)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.style.use('seaborn-v0_8-whitegrid')

CLASSES      = ['drinking', 'lecture', 'play_phone', 'sleeping', 'talking', 'watch_computer', 'writing']
CLASS_ABBREV = ['drink', 'lecture', 'phone', 'sleep', 'talk', 'watch_pc', 'write']

RESULTS = {
    'MGSAN MediaPipe (133kp)': 'MGSAN/work_dir/eduaction_kfold/mgsan_joint/kfold_summary.json',
    'MGSAN MediaPipe (127kp)': 'MGSAN/work_dir/eduaction_kfold/mgsan_upper/kfold_summary.json',
    'MGSAN MediaPipe (17kp)':  'MGSAN/work_dir/eduaction_kfold/mgsan_body17/kfold_summary.json',
    'MGSAN ViTPose (17kp)':     'MGSAN/work_dir/eduaction_kfold/mgsan_coco17/kfold_summary.json',
    'ST-GCN MediaPipe (133kp)':'st-gcn/work_dir/recognition/eduaction_kfold/ST_GCN/kfold_summary.json',
    'ST-GCN MediaPipe (127kp)':'st-gcn/work_dir/recognition/eduaction_upper_kfold/ST_GCN/kfold_summary.json',
    'ST-GCN MediaPipe (17kp)': 'st-gcn/work_dir/recognition/eduaction_body17_kfold/ST_GCN/kfold_summary.json',
    'ST-GCN ViTPose (17kp)':    'st-gcn/work_dir/recognition/eduaction_coco17_kfold/ST_GCN/kfold_summary.json',
    'DDNet MediaPipe (133kp)': 'MODEL_DDNET/work_dir/ddnet_kfold_full_body/kfold_summary.json',
    'DDNet MediaPipe (127kp)': 'MODEL_DDNET/work_dir/ddnet_kfold_upper_body/kfold_summary.json',
    'DDNet MediaPipe (17kp)':  'MODEL_DDNET/work_dir/ddnet_kfold_body_no_feet/kfold_summary.json',
    'DDNet ViTPose (17kp)':     'MODEL_DDNET/work_dir/ddnet_kfold_coco17/kfold_summary.json',
}

MODELS   = ['MGSAN', 'ST-GCN', 'DDNet']
VARIANTS = ['MediaPipe (133kp)', 'MediaPipe (127kp)', 'MediaPipe (17kp)', 'ViTPose (17kp)']

MODEL_COLORS   = {'MGSAN': '#3498db', 'ST-GCN': '#2ecc71', 'DDNet': '#9b59b6'}
VARIANT_COLORS = {
    'MediaPipe (133kp)': '#3498db',
    'MediaPipe (127kp)': '#e74c3c',
    'MediaPipe (17kp)':  '#27ae60',
    'ViTPose (17kp)':     '#f39c12',
}

def _meta(name):
    """Return (extractor, kp_count) for a model name."""
    if 'ViTPose (17kp)'      in name: return 'ViTPose',    '17'
    if 'MediaPipe (17kp)'   in name: return 'MediaPipe',  '17'
    if 'MediaPipe (127kp)'  in name: return 'MediaPipe', '127'
    return                                   'MediaPipe', '133'


def load_results(base_dir):
    results = {}
    for name, path in RESULTS.items():
        fp = os.path.join(base_dir, path)
        if os.path.exists(fp):
            with open(fp) as f:
                results[name] = json.load(f)
            print(f"  OK  {name}")
        else:
            print(f"  --  {name}")
    return results


def print_summary_table(results):
    print("\n" + "="*110)
    print("K-FOLD CROSS VALIDATION SUMMARY")
    print("="*110)
    print(f"\n{'Rank':<5} {'Model':<22} {'Extractor':<12} {'KP':>4}  {'Mean':>8} {'Std':>7} {'Min':>8} {'Max':>8} {'Time':>8}")
    print("-"*85)

    rows = []
    for name, data in results.items():
        ext, kp = _meta(name)
        accs = [r['best_val_acc'] for r in data['fold_results']]
        rows.append((name, ext, kp, data['mean_accuracy'], data['std_accuracy'],
                     min(accs), max(accs), data['total_training_time_seconds']/60))

    rows.sort(key=lambda x: -x[3])
    for i, r in enumerate(rows, 1):
        print(f"#{i:<4} {r[0]:<22} {r[1]:<12} {r[2]:>4}  {r[3]:>7.2f}%  "
              f"{r[4]:>5.2f}%  {r[5]:>6.2f}%  {r[6]:>6.2f}%  {r[7]:>6.1f}m")
    print("-"*85)
    best = rows[0]
    print(f"\nBest: {best[0]}  ->  {best[3]:.2f}% (+/-{best[4]:.2f}%)")


def plot_accuracy_comparison(results, save_path):
    """4-group bar chart per model."""
    x     = np.arange(len(MODELS))
    width = 0.20
    offsets = [-1.5, -0.5, 0.5, 1.5]

    fig, ax = plt.subplots(figsize=(14, 6))

    for vi, variant in enumerate(VARIANTS):
        means, stds = [], []
        for model in MODELS:
            key = f"{model} {variant}"
            if key in results:
                means.append(results[key]['mean_accuracy'])
                stds.append(results[key]['std_accuracy'])
            else:
                means.append(0); stds.append(0)

        bars = ax.bar(x + offsets[vi]*width, means, width, yerr=stds,
                      label=variant, color=VARIANT_COLORS[variant],
                      alpha=0.85, capsize=4)

        for bar, m, s in zip(bars, means, stds):
            if m > 0:
                ax.annotate(f'{m:.1f}\n(+/-{s:.1f})',
                            xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                            xytext=(0, 3), textcoords='offset points',
                            ha='center', va='bottom', fontsize=7.5)

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('K-Fold Accuracy Comparison — All Keypoint Variants\n'
                 '(5-Fold, seed=42, 64 frames)', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=12)
    ax.legend(title='Keypoint Set', fontsize=10)
    ax.set_ylim(55, 95)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_fold_results(results, save_path):
    """4-subplot line chart."""
    mcolors = [MODEL_COLORS[m] for m in MODELS]
    subtitles = ['MediaPipe (133kp)', 'MediaPipe (127kp)',
                 'MediaPipe (17kp)',  'ViTPose (17kp)']

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)

    for ax, variant, subtitle in zip(axes, VARIANTS, subtitles):
        for model, color in zip(MODELS, mcolors):
            key = f"{model} {variant}"
            if key in results:
                accs = [r['best_val_acc'] for r in results[key]['fold_results']]
                ax.plot(range(1, 6), accs, 'o-', label=model,
                        color=color, markersize=7, linewidth=2)
                ax.axhline(y=results[key]['mean_accuracy'], color=color,
                           linestyle='--', alpha=0.4)
        ax.set_title(subtitle, fontsize=10)
        ax.set_xlabel('Fold')
        ax.set_xticks(range(1, 6))
        ax.set_ylim(50, 100)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel('Accuracy (%)', fontsize=11)
    plt.suptitle('Per-Fold Accuracy Results', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_per_class_accuracy(results, save_path):
    mcolors = [MODEL_COLORS[m] for m in MODELS]
    subtitles = ['MediaPipe (133kp)', 'MediaPipe (127kp)',
                 'MediaPipe (17kp)',  'ViTPose (17kp)']
    x     = np.arange(len(CLASSES))
    width = 0.25

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    axes = axes.flatten()

    for ax, variant, subtitle in zip(axes, VARIANTS, subtitles):
        for i, (model, color) in enumerate(zip(MODELS, mcolors)):
            key = f"{model} {variant}"
            if key in results:
                per_class = np.array(results[key]['avg_per_class_accuracy']) * 100
                ax.bar(x + (i-1)*width, per_class, width, label=model,
                       color=color, alpha=0.8)
        ax.set_title(subtitle, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_ABBREV, rotation=45, ha='right', fontsize=9)
        ax.set_ylim(0, 115)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylabel('Accuracy (%)')

    plt.suptitle('Per-Class Accuracy (averaged across folds)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_confusion_matrices(results, save_path):
    """Best config per model family."""
    selected = []
    for model in MODELS:
        available = [f"{model} {v}" for v in VARIANTS if f"{model} {v}" in results]
        if available:
            selected.append(max(available, key=lambda x: results[x]['mean_accuracy']))

    fig, axes = plt.subplots(1, len(selected), figsize=(6*len(selected), 5))
    if len(selected) == 1:
        axes = [axes]

    for ax, name in zip(axes, selected):
        cm = np.array(results[name]['aggregated_confusion_matrix'])
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        sns.heatmap(cm_norm, annot=True, fmt='.0f', cmap='Blues', ax=ax,
                    xticklabels=CLASS_ABBREV, yticklabels=CLASS_ABBREV,
                    cbar_kws={'label': '%'})
        ax.set_title(f'{name}\n({results[name]["mean_accuracy"]:.1f}% +/- {results[name]["std_accuracy"]:.1f}%)',
                     fontsize=11)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    plt.suptitle('Aggregated Confusion Matrices — Best Config per Model', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_keypoint_comparison(results, save_path):
    """Horizontal ranked bar — all 12 models."""
    items  = sorted(results.items(), key=lambda x: x[1]['mean_accuracy'])
    names  = [k for k, _ in items]
    means  = [v['mean_accuracy'] for _, v in items]
    stds   = [v['std_accuracy']  for _, v in items]

    colors  = [MODEL_COLORS[next(m for m in MODELS if m in n)] for n in names]
    hatches = []
    for n in names:
        if 'ViTPose (17kp)'     in n: hatches.append('///')
        elif 'MediaPipe (17kp)'in n: hatches.append('xxx')
        elif '(127kp)'         in n: hatches.append('...')
        else:                        hatches.append('')

    fig, ax = plt.subplots(figsize=(11, 7))
    y    = np.arange(len(names))
    bars = ax.barh(y, means, xerr=stds, color=colors, hatch=hatches,
                   capsize=4, alpha=0.82, height=0.6)

    for bar, m, s in zip(bars, means, stds):
        ax.text(m + s + 0.2, bar.get_y() + bar.get_height()/2,
                f'{m:.1f}%', va='center', fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel('Mean Accuracy (%)', fontsize=11)
    ax.set_title('All Models Ranked by Accuracy\n'
                 'Hatch: blank=MediaPipe(133kp)  ...=MediaPipe(127kp)  xxx=MediaPipe(17kp)  ///=ViTPose(17kp)',
                 fontsize=11)
    ax.set_xlim(60, 92)
    ax.grid(axis='x', alpha=0.3)

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=MODEL_COLORS[m], label=m) for m in MODELS],
              loc='lower right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_17kp_comparison(results, save_path):
    """Special: compare 3 x 17-kp variants (body_no_feet MP vs COCO-17 VP) per model."""
    variants_17 = ['MediaPipe (17kp)', 'ViTPose (17kp)']
    colors_17   = [VARIANT_COLORS['MediaPipe (17kp)'], VARIANT_COLORS['ViTPose (17kp)']]
    x     = np.arange(len(MODELS))
    width = 0.30

    fig, ax = plt.subplots(figsize=(10, 5))

    for vi, (variant, color) in enumerate(zip(variants_17, colors_17)):
        means, stds = [], []
        for model in MODELS:
            key = f"{model} {variant}"
            if key in results:
                means.append(results[key]['mean_accuracy'])
                stds.append(results[key]['std_accuracy'])
            else:
                means.append(0); stds.append(0)

        offset = (vi - 0.5) * width
        bars = ax.bar(x + offset, means, width, yerr=stds,
                      label=variant,
                      color=color, alpha=0.85, capsize=5)

        for bar, m, s in zip(bars, means, stds):
            if m > 0:
                ax.annotate(f'{m:.1f}%\n+/-{s:.1f}',
                            xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                            xytext=(0, 3), textcoords='offset points',
                            ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('17-Joint Comparison: MediaPipe vs ViTPose\n(same skeleton, different extractor)',
                 fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=12)
    ax.legend(fontsize=10)
    ax.set_ylim(60, 95)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_training_time(results, save_path):
    fig, ax = plt.subplots(figsize=(15, 5))
    names  = list(results.keys())
    times  = [results[m]['total_training_time_seconds']/60 for m in names]
    accs   = [results[m]['mean_accuracy'] for m in names]
    colors = [MODEL_COLORS[next(m for m in MODELS if m in n)] for n in names]

    bars = ax.bar(range(len(names)), times, color=colors, alpha=0.8)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=7.5)

    ax.set_ylabel('Training Time (minutes)', fontsize=11)
    ax.set_title('Training Time — 5-Fold Total', fontsize=12)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(' MediaPipe', '\nMediaPipe').replace(' ViTPose', '\nViTPose')
                        for n in names], fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=MODEL_COLORS[m], label=m) for m in MODELS], loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def create_summary_report(results, save_path):
    rows = sorted(results.items(), key=lambda x: -x[1]['mean_accuracy'])

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("K-FOLD CROSS VALIDATION SUMMARY REPORT\n")
        f.write("EduAction Dataset -- Action Recognition\n")
        f.write("="*80 + "\n\n")
        f.write("Settings: 5-Fold | seed=42 | window=64 frames | 7 classes | ~320 videos\n\n")

        f.write(f"{'Rank':<5} {'Model':<22} {'Extractor':<12} {'KP':>4}  {'Mean':>8} {'Std':>7} {'Time':>7}\n")
        f.write("-"*70 + "\n")
        for rank, (name, data) in enumerate(rows, 1):
            ext, kp = _meta(name)
            t = data['total_training_time_seconds'] / 60
            f.write(f"#{rank:<4} {name:<22} {ext:<12} {kp:>4}  "
                    f"{data['mean_accuracy']:>7.2f}%  {data['std_accuracy']:>5.2f}%  {t:>5.1f}m\n")

        best = rows[0]
        f.write(f"\nBest Overall: {best[0]}  ->  {best[1]['mean_accuracy']:.2f}% (+/-{best[1]['std_accuracy']:.2f}%)\n")

        # Per-model best
        f.write("\n\nBEST PER MODEL:\n" + "-"*40 + "\n")
        for model in MODELS:
            variants = [(f"{model} {v}", results[f"{model} {v}"]) for v in VARIANTS if f"{model} {v}" in results]
            best_v = max(variants, key=lambda x: x[1]['mean_accuracy'])
            f.write(f"  {model:<8}: {best_v[0]:<22}  {best_v[1]['mean_accuracy']:.2f}%\n")

        # 17-kp extractor comparison
        f.write("\n\n17-JOINT COMPARISON (MediaPipe vs ViTPose):\n" + "-"*50 + "\n")
        for model in MODELS:
            mp  = results.get(f"{model} Body-17")
            vp  = results.get(f"{model} COCO-17")
            if mp and vp:
                diff = mp['mean_accuracy'] - vp['mean_accuracy']
                winner = "MediaPipe" if diff > 0 else "ViTPose"
                f.write(f"  {model:<8}: MP={mp['mean_accuracy']:.2f}%  VP={vp['mean_accuracy']:.2f}%"
                        f"  diff={diff:+.2f}%  winner={winner}\n")

        # Per-class of best overall
        f.write(f"\n\nPER-CLASS -- {best[0]}:\n" + "-"*40 + "\n")
        per_class = best[1]['avg_per_class_accuracy']
        for cls, acc in sorted(zip(CLASSES, per_class), key=lambda x: -x[1]):
            bar = '#' * int(acc * 20)
            f.write(f"  {cls:<20}: {acc*100:>5.1f}%  {bar}\n")

        sorted_cls = sorted(zip(CLASSES, per_class), key=lambda x: x[1])
        f.write(f"\n  Hardest: {sorted_cls[0][0]} ({sorted_cls[0][1]*100:.1f}%)\n")
        f.write(f"  Easiest: {sorted_cls[-1][0]} ({sorted_cls[-1][1]*100:.1f}%)\n")

    print(f"Saved: {save_path}")


def main():
    base_dir   = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'kfold_visualization')
    os.makedirs(output_dir, exist_ok=True)

    print("Loading results...")
    results = load_results(base_dir)
    if not results:
        print("No results found!")
        return

    print_summary_table(results)
    print("\nGenerating visualizations...")

    plot_accuracy_comparison(results, os.path.join(output_dir, '1_accuracy_comparison.png'))
    plot_fold_results(results,        os.path.join(output_dir, '2_fold_results.png'))
    plot_per_class_accuracy(results,  os.path.join(output_dir, '3_per_class_accuracy.png'))
    plot_confusion_matrices(results,  os.path.join(output_dir, '4_confusion_matrices.png'))
    plot_keypoint_comparison(results, os.path.join(output_dir, '5_keypoint_comparison.png'))
    plot_17kp_comparison(results,     os.path.join(output_dir, '6_mediapipe_vs_vipose_17kp.png'))
    plot_training_time(results,       os.path.join(output_dir, '7_training_time.png'))
    create_summary_report(results,    os.path.join(output_dir, 'summary_report.txt'))

    print(f"\nDone -> {output_dir}")


if __name__ == '__main__':
    main()
