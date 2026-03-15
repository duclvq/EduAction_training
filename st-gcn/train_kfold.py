#!/usr/bin/env python
"""
K-Fold Cross Validation Training Script for ST-GCN on EduAction dataset.

Features:
- Stratified K-Fold with deterministic splitting
- All random seeds fixed for reproducibility
- Automatic result aggregation across folds
- Confusion matrix and per-class accuracy tracking

Usage:
    python train_kfold.py --config config/st_gcn/eduaction/train_kfold.yaml
    python train_kfold.py --config config/st_gcn/eduaction/train_kfold_upper.yaml
"""

import os
import sys
import argparse
import yaml
import json
import random
import numpy as np
import torch


def _worker_init_fn(worker_id):
    """Module-level worker init for Windows multiprocessing compatibility."""
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import defaultdict
import time
from datetime import datetime

# Add st-gcn root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from net.st_gcn import Model
from feeder.feeder_eduaction_kfold import FeederEduActionKFold, set_all_seeds


def weights_init(m):
    """Initialize weights with fixed seed."""
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class KFoldTrainer:
    """K-Fold Cross Validation Trainer for ST-GCN."""

    def __init__(self, config_path):
        self.load_config(config_path)
        self.setup_environment()

    def load_config(self, config_path):
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Set defaults
        self.config.setdefault('n_folds', 5)
        self.config.setdefault('seed', 42)
        self.config.setdefault('num_epoch', 120)
        self.config.setdefault('batch_size', 16)
        self.config.setdefault('test_batch_size', 16)
        self.config.setdefault('base_lr', 0.01)
        self.config.setdefault('weight_decay', 0.0001)
        self.config.setdefault('step', [40, 60, 80])
        self.config.setdefault('optimizer', 'SGD')
        self.config.setdefault('nesterov', True)
        self.config.setdefault('num_worker', 4)
        self.config.setdefault('eval_interval', 5)
        self.config.setdefault('save_interval', 10)
        self.config.setdefault('window_size', 64)
        self.config.setdefault('keypoint_subset', 'full_body')
        self.config.setdefault('random_choose', True)
        self.config.setdefault('random_move', True)

        # Model args defaults
        model_args = self.config.get('model_args', {})
        model_args.setdefault('in_channels', 2)
        model_args.setdefault('num_class', 7)
        model_args.setdefault('edge_importance_weighting', True)
        model_args.setdefault('dropout', 0.5)
        self.config['model_args'] = model_args

        # Graph args
        graph_args = model_args.get('graph_args', {})
        graph_args.setdefault('layout', 'eduaction')
        graph_args.setdefault('strategy', 'spatial')
        model_args['graph_args'] = graph_args

        self.config_path = config_path

    def setup_environment(self):
        """Setup work directory and device."""
        self.work_dir = self.config.get('work_dir', './work_dir/kfold')
        os.makedirs(self.work_dir, exist_ok=True)

        # Save config
        config_save_path = os.path.join(self.work_dir, 'config.yaml')
        with open(config_save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

        # Setup device
        self.use_gpu = self.config.get('use_gpu', True) and torch.cuda.is_available()
        if self.use_gpu:
            device_ids = self.config.get('device', [0])
            self.device = torch.device(f'cuda:{device_ids[0]}')
        else:
            self.device = torch.device('cpu')

        print(f"Using device: {self.device}")

        # Class names
        self.classes = ['drinking', 'lecture', 'play_phone', 'sleeping',
                       'talking', 'watch_computer', 'writing']

    def create_model(self, fold_seed):
        """Create and initialize model with deterministic weights."""
        set_all_seeds(fold_seed)

        model = Model(**self.config['model_args'])
        model.apply(weights_init)
        model = model.to(self.device)

        return model

    def create_dataloaders(self, fold_idx):
        """Create train and validation dataloaders for a fold."""
        data_dir = self.config['data_dir']

        train_dataset = FeederEduActionKFold(
            data_dir=data_dir,
            split='train',
            n_folds=self.config['n_folds'],
            fold_idx=fold_idx,
            seed=self.config['seed'],
            keypoint_subset=self.config['keypoint_subset'],
            window_size=self.config['window_size'],
            random_choose=self.config['random_choose'],
            random_move=self.config['random_move'],
        )

        val_dataset = FeederEduActionKFold(
            data_dir=data_dir,
            split='val',
            n_folds=self.config['n_folds'],
            fold_idx=fold_idx,
            seed=self.config['seed'],
            keypoint_subset=self.config['keypoint_subset'],
            window_size=self.config['window_size'],
            random_choose=False,
            random_move=False,
        )

        # Use generator for reproducible shuffling
        g = torch.Generator()
        g.manual_seed(self.config['seed'] + fold_idx)

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_worker'],
            drop_last=True,
            worker_init_fn=_worker_init_fn,
            generator=g,
        )

        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.config['test_batch_size'],
            shuffle=False,
            num_workers=self.config['num_worker'],
        )

        return train_loader, val_loader

    def create_optimizer(self, model):
        """Create optimizer."""
        if self.config['optimizer'] == 'SGD':
            optimizer = optim.SGD(
                model.parameters(),
                lr=self.config['base_lr'],
                momentum=0.9,
                nesterov=self.config['nesterov'],
                weight_decay=self.config['weight_decay']
            )
        elif self.config['optimizer'] == 'Adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.config['base_lr'],
                weight_decay=self.config['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config['optimizer']}")

        return optimizer

    def adjust_lr(self, optimizer, epoch):
        """Adjust learning rate based on step schedule."""
        step = np.array(self.config['step'])
        lr = self.config['base_lr'] * (0.1 ** np.sum(epoch >= step))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def train_epoch(self, model, train_loader, optimizer, criterion, epoch):
        """Train for one epoch."""
        model.train()
        lr = self.adjust_lr(optimizer, epoch)

        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, label) in enumerate(train_loader):
            data = data.float().to(self.device)
            label = label.long().to(self.device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy, lr

    def validate(self, model, val_loader, criterion):
        """Validate the model."""
        model.eval()

        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data, label in val_loader:
                data = data.float().to(self.device)
                label = label.long().to(self.device)

                output = model(data)
                loss = criterion(output, label)

                total_loss += loss.item()
                _, predicted = output.max(1)
                total += label.size(0)
                correct += predicted.eq(label).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(label.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total

        # Compute confusion matrix
        num_classes = len(self.classes)
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
        for pred, label in zip(all_preds, all_labels):
            confusion_matrix[label][pred] += 1

        # Per-class accuracy
        per_class_acc = []
        for i in range(num_classes):
            if confusion_matrix[i].sum() > 0:
                per_class_acc.append(confusion_matrix[i][i] / confusion_matrix[i].sum())
            else:
                per_class_acc.append(0.0)

        return avg_loss, accuracy, confusion_matrix, per_class_acc

    def train_fold(self, fold_idx, log_file):
        """Train a single fold."""
        print(f"\n{'='*60}")
        print(f"Training Fold {fold_idx + 1}/{self.config['n_folds']}")
        print(f"{'='*60}")

        # Set seed for this fold
        fold_seed = self.config['seed'] + fold_idx * 1000
        set_all_seeds(fold_seed)

        # Create model, data, optimizer
        model = self.create_model(fold_seed)
        train_loader, val_loader = self.create_dataloaders(fold_idx)
        optimizer = self.create_optimizer(model)
        criterion = nn.CrossEntropyLoss()

        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': [],
        }

        best_val_acc = 0
        best_epoch = 0
        best_confusion_matrix = None
        best_per_class_acc = None

        # Create fold directory
        fold_dir = os.path.join(self.work_dir, f'fold_{fold_idx + 1}')
        os.makedirs(fold_dir, exist_ok=True)

        for epoch in range(self.config['num_epoch']):
            # Update augmentation seed per epoch for varied but reproducible augmentation
            train_loader.dataset.set_augment_seed(fold_seed + epoch * 100)

            # Train
            train_loss, train_acc, lr = self.train_epoch(
                model, train_loader, optimizer, criterion, epoch
            )

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['lr'].append(lr)

            # Validate
            if (epoch + 1) % self.config['eval_interval'] == 0 or epoch == self.config['num_epoch'] - 1:
                val_loss, val_acc, confusion_matrix, per_class_acc = self.validate(
                    model, val_loader, criterion
                )

                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)

                log_msg = (f"[Fold {fold_idx + 1}] Epoch {epoch + 1:3d} | "
                          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {lr:.6f}")
                print(log_msg)
                log_file.write(log_msg + '\n')
                log_file.flush()

                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch + 1
                    best_confusion_matrix = confusion_matrix
                    best_per_class_acc = per_class_acc
                    torch.save(model.state_dict(), os.path.join(fold_dir, 'best_model.pt'))

            # Save checkpoint periodically
            if (epoch + 1) % self.config['save_interval'] == 0:
                torch.save(model.state_dict(), os.path.join(fold_dir, f'epoch{epoch + 1}_model.pt'))

        # Save final model and history
        torch.save(model.state_dict(), os.path.join(fold_dir, 'final_model.pt'))

        with open(os.path.join(fold_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)

        # Save best results
        fold_result = {
            'fold_idx': fold_idx,
            'best_val_acc': best_val_acc,
            'best_epoch': best_epoch,
            'confusion_matrix': best_confusion_matrix.tolist(),
            'per_class_acc': best_per_class_acc,
        }

        with open(os.path.join(fold_dir, 'best_result.json'), 'w') as f:
            json.dump(fold_result, f, indent=2)

        print(f"\nFold {fold_idx + 1} Best: {best_val_acc:.2f}% at epoch {best_epoch}")

        return fold_result

    def run(self):
        """Run K-Fold cross validation."""
        print(f"\n{'#'*60}")
        print(f"K-Fold Cross Validation Training")
        print(f"{'#'*60}")
        print(f"Config: {self.config_path}")
        print(f"Work dir: {self.work_dir}")
        print(f"Folds: {self.config['n_folds']}")
        print(f"Seed: {self.config['seed']}")
        print(f"Keypoints: {self.config['keypoint_subset']}")
        print(f"Epochs: {self.config['num_epoch']}")

        # Open log file
        log_path = os.path.join(self.work_dir, 'training_log.txt')
        log_file = open(log_path, 'w')
        log_file.write(f"Training started: {datetime.now()}\n")
        log_file.write(f"Config: {json.dumps(self.config, indent=2)}\n\n")

        # Train all folds
        all_results = []
        start_time = time.time()

        for fold_idx in range(self.config['n_folds']):
            fold_result = self.train_fold(fold_idx, log_file)
            all_results.append(fold_result)

        total_time = time.time() - start_time

        # Aggregate results
        all_val_accs = [r['best_val_acc'] for r in all_results]
        mean_acc = np.mean(all_val_accs)
        std_acc = np.std(all_val_accs)

        # Aggregate confusion matrices
        total_cm = np.zeros((len(self.classes), len(self.classes)), dtype=int)
        for r in all_results:
            total_cm += np.array(r['confusion_matrix'])

        # Average per-class accuracy
        avg_per_class_acc = np.mean([r['per_class_acc'] for r in all_results], axis=0)

        # Print summary
        summary = f"""
{'='*60}
K-FOLD CROSS VALIDATION SUMMARY
{'='*60}
Folds: {self.config['n_folds']}
Seed: {self.config['seed']}
Keypoint subset: {self.config['keypoint_subset']}
Total training time: {total_time/60:.1f} minutes

FOLD RESULTS:
"""
        for i, r in enumerate(all_results):
            summary += f"  Fold {i + 1}: {r['best_val_acc']:.2f}% (epoch {r['best_epoch']})\n"

        summary += f"""
OVERALL:
  Mean Accuracy: {mean_acc:.2f}% (+/- {std_acc:.2f}%)
  Min: {min(all_val_accs):.2f}%
  Max: {max(all_val_accs):.2f}%

PER-CLASS ACCURACY (averaged across folds):
"""
        for i, (cls, acc) in enumerate(zip(self.classes, avg_per_class_acc)):
            summary += f"  {cls:20s}: {acc*100:.2f}%\n"

        summary += f"""
AGGREGATED CONFUSION MATRIX:
{'':20s} """ + " ".join(f"{c[:6]:>6s}" for c in self.classes) + "\n"

        for i, row in enumerate(total_cm):
            summary += f"{self.classes[i]:20s} " + " ".join(f"{v:>6d}" for v in row) + "\n"

        summary += f"\n{'='*60}\n"

        print(summary)
        log_file.write(summary)
        log_file.close()

        # Save final summary
        final_summary = {
            'n_folds': self.config['n_folds'],
            'seed': self.config['seed'],
            'keypoint_subset': self.config['keypoint_subset'],
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'fold_results': all_results,
            'aggregated_confusion_matrix': total_cm.tolist(),
            'avg_per_class_accuracy': avg_per_class_acc.tolist(),
            'total_training_time_seconds': total_time,
        }

        with open(os.path.join(self.work_dir, 'kfold_summary.json'), 'w') as f:
            json.dump(final_summary, f, indent=2)

        print(f"\nResults saved to: {self.work_dir}")

        return final_summary


def main():
    parser = argparse.ArgumentParser(description='K-Fold Cross Validation for ST-GCN')
    parser.add_argument('--config', '-c', required=True, help='Path to config file')
    args = parser.parse_args()

    trainer = KFoldTrainer(args.config)
    trainer.run()


if __name__ == '__main__':
    main()
