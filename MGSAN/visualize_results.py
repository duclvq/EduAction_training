"""
Visualization script for MGSAN training results on EduAction dataset.
Generates confusion matrix, accuracy plots, per-class metrics, and more.
"""
import os
import sys
import argparse
import numpy as np
import torch
import yaml
import pickle
from collections import OrderedDict
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Class names for EduAction
CLASS_NAMES = ['drinking', 'lecture', 'play_phone', 'sleeping', 'talking', 'watch_computer', 'writing']


def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    return getattr(sys.modules[mod_str], class_str)


def load_model(config_path, weights_path):
    """Load model from config and weights."""
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    Model = import_class(config['model'])
    model = Model(**config['model_args'])

    weights = torch.load(weights_path, map_location='cpu')
    weights = OrderedDict([[k.split('module.')[-1], v] for k, v in weights.items()])
    model.load_state_dict(weights)

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()
    return model, config


def get_predictions(model, data_loader):
    """Get predictions from model."""
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for data, label, index in tqdm(data_loader, desc="Evaluating"):
            if torch.cuda.is_available():
                data = data.float().cuda()
            else:
                data = data.float()

            output = model(data)
            probs = torch.softmax(output, dim=1)
            _, preds = torch.max(output, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path, title='Confusion Matrix'):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title(f'{title} (Counts)')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].tick_params(axis='y', rotation=0)

    # Normalized (percentages)
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title(f'{title} (Normalized)')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].tick_params(axis='y', rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

    return cm


def plot_per_class_metrics(y_true, y_pred, class_names, save_path):
    """Plot per-class precision, recall, F1-score."""
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#2ecc71')
    bars2 = ax.bar(x, recall, width, label='Recall', color='#3498db')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c')

    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Per-class metrics saved to {save_path}")


def plot_class_accuracy(y_true, y_pred, class_names, save_path):
    """Plot per-class accuracy as horizontal bar chart."""
    cm = confusion_matrix(y_true, y_pred)
    class_acc = np.diag(cm) / cm.sum(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.RdYlGn(class_acc)
    bars = ax.barh(class_names, class_acc, color=colors)

    ax.set_xlabel('Accuracy')
    ax.set_title('Per-Class Accuracy')
    ax.set_xlim(0, 1)
    ax.axvline(x=np.mean(class_acc), color='red', linestyle='--', label=f'Mean: {np.mean(class_acc):.2%}')
    ax.legend()

    # Add value labels
    for bar, acc in zip(bars, class_acc):
        ax.text(acc + 0.02, bar.get_y() + bar.get_height()/2,
                f'{acc:.2%}', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Class accuracy saved to {save_path}")


def plot_prediction_distribution(y_true, y_pred, class_names, save_path):
    """Plot distribution of predictions vs actual labels."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Actual distribution
    unique, counts = np.unique(y_true, return_counts=True)
    axes[0].bar(class_names, counts, color='#3498db')
    axes[0].set_title('Actual Label Distribution')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)

    # Predicted distribution
    unique, counts = np.unique(y_pred, return_counts=True)
    pred_counts = np.zeros(len(class_names))
    for u, c in zip(unique, counts):
        pred_counts[u] = c
    axes[1].bar(class_names, pred_counts, color='#e74c3c')
    axes[1].set_title('Predicted Label Distribution')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Count')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Prediction distribution saved to {save_path}")


def plot_confidence_distribution(probs, y_true, y_pred, save_path):
    """Plot confidence distribution for correct vs incorrect predictions."""
    max_probs = np.max(probs, axis=1)
    correct = y_true == y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(max_probs[correct], bins=20, alpha=0.7, label='Correct', color='#2ecc71')
    axes[0].hist(max_probs[~correct], bins=20, alpha=0.7, label='Incorrect', color='#e74c3c')
    axes[0].set_xlabel('Confidence')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Confidence Distribution')
    axes[0].legend()

    # Box plot
    data = [max_probs[correct], max_probs[~correct]]
    bp = axes[1].boxplot(data, labels=['Correct', 'Incorrect'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#2ecc71')
    bp['boxes'][1].set_facecolor('#e74c3c')
    axes[1].set_ylabel('Confidence')
    axes[1].set_title('Confidence by Prediction Correctness')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confidence distribution saved to {save_path}")


def generate_report(y_true, y_pred, class_names, save_path):
    """Generate text report with all metrics."""
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)

    accuracy = np.mean(y_true == y_pred)

    with open(save_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("MGSAN EduAction Evaluation Report\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")

        f.write("Classification Report:\n")
        f.write("-" * 60 + "\n")
        f.write(report)
        f.write("\n")

        # Per-class accuracy
        cm = confusion_matrix(y_true, y_pred)
        class_acc = np.diag(cm) / cm.sum(axis=1)

        f.write("\nPer-Class Accuracy:\n")
        f.write("-" * 60 + "\n")
        for name, acc in zip(class_names, class_acc):
            f.write(f"  {name:20s}: {acc:.4f} ({acc*100:.2f}%)\n")

        f.write("\n" + "=" * 60 + "\n")

    print(f"Report saved to {save_path}")
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description='Visualize MGSAN results')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--weights', required=True, help='Path to model weights')
    parser.add_argument('--output-dir', default='./visualization_results', help='Output directory')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Load model
    print("Loading model...")
    model, _ = load_model(args.config, args.weights)

    # Load test data
    print("Loading test data...")
    Feeder = import_class(config['feeder'])
    test_dataset = Feeder(**config['test_feeder_args'])
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Get predictions
    print("Running evaluation...")
    y_pred, y_true, probs = get_predictions(model, test_loader)

    # Generate visualizations
    print("\nGenerating visualizations...")

    # 1. Confusion Matrix
    plot_confusion_matrix(
        y_true, y_pred, CLASS_NAMES,
        os.path.join(args.output_dir, 'confusion_matrix.png')
    )

    # 2. Per-class metrics
    plot_per_class_metrics(
        y_true, y_pred, CLASS_NAMES,
        os.path.join(args.output_dir, 'per_class_metrics.png')
    )

    # 3. Class accuracy
    plot_class_accuracy(
        y_true, y_pred, CLASS_NAMES,
        os.path.join(args.output_dir, 'class_accuracy.png')
    )

    # 4. Prediction distribution
    plot_prediction_distribution(
        y_true, y_pred, CLASS_NAMES,
        os.path.join(args.output_dir, 'prediction_distribution.png')
    )

    # 5. Confidence distribution
    plot_confidence_distribution(
        probs, y_true, y_pred,
        os.path.join(args.output_dir, 'confidence_distribution.png')
    )

    # 6. Text report
    generate_report(
        y_true, y_pred, CLASS_NAMES,
        os.path.join(args.output_dir, 'evaluation_report.txt')
    )

    print(f"\nAll visualizations saved to {args.output_dir}")


if __name__ == '__main__':
    main()
