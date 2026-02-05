#!/usr/bin/env python
"""
K-Fold Cross Validation Training Script for MGSAN on EduAction dataset.

Features:
- Stratified K-Fold with deterministic splitting
- All random seeds fixed for reproducibility (Python, NumPy, PyTorch)
- Automatic result aggregation across folds
- Confusion matrix and per-class accuracy tracking

Usage:
    python train_kfold.py --config config/eduaction/kfold.yaml
    python train_kfold.py --config config/eduaction/kfold_upper.yaml
"""

import os
import sys
import argparse
import yaml
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import defaultdict
import time
from datetime import datetime
import csv

# Add MGSAN root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Use CUDA 12.x compatible model for RTX 50 series (Blackwell architecture)
from model.MGSAN_compat import Model
from feeders.feeder_eduaction_kfold import FeederEduActionKFold, set_all_seeds


def import_class(name):
    """Dynamically import a class from a module."""
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def weights_init(m):
    """Initialize weights with deterministic values."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


class KFoldTrainer:
    """K-Fold Cross Validation Trainer for MGSAN."""

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
        self.config.setdefault('base_lr', 0.1)
        self.config.setdefault('weight_decay', 0.0001)
        self.config.setdefault('step', [60, 80, 100])
        self.config.setdefault('optimizer', 'SGD')
        self.config.setdefault('nesterov', True)
        self.config.setdefault('num_worker', 4)
        self.config.setdefault('eval_interval', 1)
        self.config.setdefault('save_interval', 10)
        self.config.setdefault('window_size', 64)
        self.config.setdefault('keypoint_subset', 'full_body')
        self.config.setdefault('p_interval', [0.5, 1])
        self.config.setdefault('random_rot', True)

        # Model args defaults
        model_args = self.config.get('model_args', {})
        model_args.setdefault('num_class', 7)
        model_args.setdefault('num_point', 133)
        model_args.setdefault('num_person', 1)
        model_args.setdefault('in_channels', 2)
        model_args.setdefault('dropout', 0.0)
        self.config['model_args'] = model_args

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
            # CUDA 12.x compatibility settings
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.enabled = True  # Keep enabled for performance
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(device_ids[0])}")
        else:
            self.device = torch.device('cpu')

        print(f"Using device: {self.device}")

        # Class names
        self.classes = ['drinking', 'lecture', 'play_phone', 'sleeping',
                       'talking', 'watch_computer', 'writing']

    def create_model(self, fold_seed):
        """Create and initialize model with deterministic weights."""
        set_all_seeds(fold_seed)

        model_args = self.config['model_args'].copy()
        model = Model(**model_args)
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
            p_interval=self.config['p_interval'],
            random_rot=self.config['random_rot'],
        )

        val_dataset = FeederEduActionKFold(
            data_dir=data_dir,
            split='val',
            n_folds=self.config['n_folds'],
            fold_idx=fold_idx,
            seed=self.config['seed'],
            keypoint_subset=self.config['keypoint_subset'],
            window_size=self.config['window_size'],
            p_interval=[1.0],  # No random crop for validation
            random_rot=False,
        )

        # Use generator for reproducible shuffling
        g = torch.Generator()
        g.manual_seed(self.config['seed'] + fold_idx)

        def worker_init_fn(worker_id):
            np.random.seed(self.config['seed'] + fold_idx + worker_id)
            random.seed(self.config['seed'] + fold_idx + worker_id)

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_worker'],
            drop_last=True,
            worker_init_fn=worker_init_fn,
            generator=g,
            pin_memory=True,  # Faster GPU transfer for CUDA 12.x
        )

        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.config['test_batch_size'],
            shuffle=False,
            num_workers=self.config['num_worker'],
            pin_memory=True,  # Faster GPU transfer for CUDA 12.x
        )

        return train_loader, val_loader, train_dataset, val_dataset

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

        for batch_idx, (data, label, index) in enumerate(train_loader):
            data = data.float().to(self.device, non_blocking=True)
            label = label.long().to(self.device, non_blocking=True)

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
            for data, label, index in val_loader:
                data = data.float().to(self.device, non_blocking=True)
                label = label.long().to(self.device, non_blocking=True)

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
        train_loader, val_loader, train_dataset, val_dataset = self.create_dataloaders(fold_idx)
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
            # Update augmentation seed per epoch
            train_dataset.set_augment_seed(fold_seed + epoch * 100)

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

                # Save per-class accuracy CSV
                csv_path = os.path.join(fold_dir, f'epoch{epoch + 1}_each_class_acc.csv')
                with open(csv_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(per_class_acc)
                    for row in confusion_matrix:
                        writer.writerow(row)

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
        print(f"K-Fold Cross Validation Training - MGSAN")
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
K-FOLD CROSS VALIDATION SUMMARY - MGSAN
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
    parser = argparse.ArgumentParser(description='K-Fold Cross Validation for MGSAN')
    parser.add_argument('--config', '-c', required=True, help='Path to config file')
    args = parser.parse_args()

    trainer = KFoldTrainer(args.config)
    trainer.run()


if __name__ == '__main__':
    main()
