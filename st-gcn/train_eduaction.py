#!/usr/bin/env python
"""
Training script for ST-GCN on EduAction dataset.
This script provides a convenient wrapper for training with various configurations.

Usage:
    python train_eduaction.py --config config/st_gcn/eduaction/train.yaml
    python train_eduaction.py --config config/st_gcn/eduaction/train_body.yaml
    python train_eduaction.py --config config/st_gcn/eduaction/train_hands.yaml
"""

import os
import sys
import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(description='Train ST-GCN on EduAction dataset')
    parser.add_argument('--config', type=str, 
                        default='config/st_gcn/eduaction/train.yaml',
                        help='Path to config file')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Override data directory from config')
    parser.add_argument('--work_dir', type=str, default=None,
                        help='Override work directory from config')
    parser.add_argument('--device', type=int, nargs='+', default=None,
                        help='GPU device IDs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--num_epoch', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--base_lr', type=float, default=None,
                        help='Override base learning rate')
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to pretrained weights for fine-tuning')
    parser.add_argument('--phase', type=str, default='train',
                        choices=['train', 'test'],
                        help='Training or testing phase')
    
    args = parser.parse_args()
    
    # Build command
    cmd = [
        sys.executable, 'main.py', 'recognition',
        '-c', args.config
    ]
    
    # Add optional overrides
    if args.data_dir:
        cmd.extend(['--train_feeder_args', f'data_dir={args.data_dir}'])
        cmd.extend(['--test_feeder_args', f'data_dir={args.data_dir}'])
    
    if args.work_dir:
        cmd.extend(['--work_dir', args.work_dir])
    
    if args.device:
        cmd.extend(['--device', *[str(d) for d in args.device]])
    
    if args.batch_size:
        cmd.extend(['--batch_size', str(args.batch_size)])
        cmd.extend(['--test_batch_size', str(args.batch_size)])
    
    if args.num_epoch:
        cmd.extend(['--num_epoch', str(args.num_epoch)])
    
    if args.base_lr:
        cmd.extend(['--base_lr', str(args.base_lr)])
    
    if args.weights:
        cmd.extend(['--weights', args.weights])
    
    if args.phase == 'test':
        cmd.extend(['--phase', 'test'])
    
    print(f"Running command: {' '.join(cmd)}")
    print("-" * 60)
    
    # Run the training
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)) or '.')
    
    return result.returncode


if __name__ == '__main__':
    sys.exit(main())
