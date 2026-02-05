#!/usr/bin/env python
"""
MGSAN Training Script - Compatible with PyTorch 2.x and CUDA 12.x

Changes from original main.py:
1. Removed 'resource' module (not available on Windows)
2. Enable cuDNN with deterministic mode (faster than disabled)
3. Use torch.device for device-agnostic code
4. Better error handling for CUDA operations
"""
from __future__ import print_function

import argparse
import inspect
import os
import pickle
import random
import shutil
import sys
import time
from collections import OrderedDict
import traceback
from sklearn.metrics import confusion_matrix
import csv
import numpy as np
import glob
import platform

# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm

from torchlight.torchlight import DictAction
from math import cos, pi

# Resource limit - only on Unix
if platform.system() != 'Windows':
    try:
        import resource
        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
    except Exception as e:
        print(f"Warning: Could not set resource limit: {e}")


def init_seed(seed):
    """Initialize random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # For CUDA 12.x compatibility, use deterministic mode but keep cuDNN enabled
        # This is faster than completely disabling cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Enable cuDNN (disabled in original causes slow performance)
        torch.backends.cudnn.enabled = True


def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def get_parser():
    parser = argparse.ArgumentParser(description='MGSAN - CUDA 12.x Compatible')
    parser.add_argument('--work-dir', default='./work_dir/temp', help='work folder for storing results')
    parser.add_argument('-model_saved_name', default='MGSAN')
    parser.add_argument('--config', default='./config/eduaction/default.yaml', help='path to config file')

    # processor
    parser.add_argument('--phase', default='train', help='must be train or test')
    parser.add_argument('--save-score', type=str2bool, default=False)

    # debug
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--save-interval', type=int, default=1)
    parser.add_argument('--save-epoch', type=int, default=60)
    parser.add_argument('--eval-interval', type=int, default=1)
    parser.add_argument('--print-log', type=str2bool, default=True)
    parser.add_argument('--show-topk', type=int, default=[1, 5], nargs='+')

    # feeder
    parser.add_argument('--feeder', default='feeders.feeder_eduaction.FeederEduAction')
    parser.add_argument('--num-worker', type=int, default=4)
    parser.add_argument('--train-feeder-args', action=DictAction, default=dict())
    parser.add_argument('--test-feeder-args', action=DictAction, default=dict())

    # model
    parser.add_argument('--model', default='model.MGSAN_compat.Model', help='model class')
    parser.add_argument('--model-args', action=DictAction, default=dict())
    parser.add_argument('--weights', default=None, help='weights for initialization')
    parser.add_argument('--ignore-weights', type=str, default=[], nargs='+')

    # optimizer
    parser.add_argument('--base-lr', type=float, default=0.1)
    parser.add_argument('--step', type=int, default=[60, 80, 100], nargs='+')
    parser.add_argument('--device', type=int, default=[0], nargs='+')
    parser.add_argument('--optimizer', default='SGD')
    parser.add_argument('--nesterov', type=str2bool, default=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--test-batch-size', type=int, default=16)
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--num-epoch', type=int, default=120)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--lr-decay-rate', type=float, default=0.1)
    parser.add_argument('--warm_up_epoch', type=int, default=5)

    # Auto-confirm for non-interactive mode
    parser.add_argument('--auto-confirm', type=str2bool, default=False, help='auto confirm directory deletion')

    return parser


class Processor:
    """Processor for Skeleton-based Action Recognition - CUDA 12.x Compatible."""

    def __init__(self, arg):
        self.arg = arg
        self.device = self._get_device()
        self.save_arg()

        if arg.phase == 'train':
            if not arg.train_feeder_args.get('debug', False):
                arg.model_saved_name = os.path.join(arg.work_dir, 'runs')
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir:', arg.model_saved_name, 'already exist')
                    if arg.auto_confirm:
                        answer = 'y'
                    else:
                        answer = input('delete it? y/n: ')
                    if answer == 'y':
                        shutil.rmtree(arg.model_saved_name)
                        print('Dir removed:', arg.model_saved_name)
                    else:
                        print('Dir not removed:', arg.model_saved_name)
                self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test'), 'test')

        self.global_step = 0
        self.load_model()

        if self.arg.phase != 'model_size':
            self.load_optimizer()
            self.load_data()

        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_acc_epoch = 0

    def _get_device(self):
        """Get the appropriate device."""
        if torch.cuda.is_available():
            device_id = self.arg.device[0] if isinstance(self.arg.device, list) else self.arg.device
            device = torch.device(f'cuda:{device_id}')
            print(f"Using CUDA device: {torch.cuda.get_device_name(device_id)}")
            print(f"CUDA version: {torch.version.cuda}")
        else:
            device = torch.device('cpu')
            print("CUDA not available, using CPU")
        return device

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()

        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                pin_memory=True,  # Faster data transfer to GPU
                worker_init_fn=lambda w: np.random.seed(self.arg.seed + w))

        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            pin_memory=True)

    def load_model(self):
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)

        self.model = Model(**self.arg.model_args)
        self.model = self.model.to(self.device)
        self.loss = nn.CrossEntropyLoss().to(self.device)

        if self.arg.weights:
            self.global_step = int(self.arg.weights[:-3].split('-')[-1])
            self.print_log(f'Load weights from {self.arg.weights}')

            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'rb') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights, map_location=self.device)

            weights = OrderedDict([[k.split('module.')[-1], v] for k, v in weights.items()])

            for w in self.arg.ignore_weights:
                for key in list(weights.keys()):
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log(f'Removed weights: {key}')

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Cannot find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

        # Multi-GPU support
        if isinstance(self.arg.device, list) and len(self.arg.device) > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.arg.device)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {self.arg.optimizer}")

        self.print_log(f'Using warm up, epoch: {self.arg.warm_up_epoch}')

    def save_arg(self):
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open(f'{self.arg.work_dir}/config.yaml', 'w') as f:
            f.write(f"# command line: {' '.join(sys.argv)}\n\n")
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
        if epoch < self.arg.warm_up_epoch:
            lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
        else:
            lr = self.arg.base_lr * (self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step)))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def print_log(self, msg, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            msg = f"[ {localtime} ] {msg}"
        print(msg)
        if self.arg.print_log:
            with open(f'{self.arg.work_dir}/log.txt', 'a') as f:
                print(msg, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, save_model=False):
        self.model.train()
        self.print_log(f'Training epoch: {epoch + 1}')
        loader = self.data_loader['train']

        self.adjust_learning_rate(epoch)

        loss_value = []
        acc_value = []
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader, ncols=80)

        for batch_idx, (data, label, index) in enumerate(process):
            self.global_step += 1

            data = data.float().to(self.device, non_blocking=True)
            label = label.long().to(self.device, non_blocking=True)
            timer['dataloader'] += self.split_time()

            output = self.model(data)
            loss = self.loss(output, label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_value.append(loss.item())
            timer['model'] += self.split_time()

            _, predict_label = torch.max(output.data, 1)
            acc = (predict_label == label.data).float().mean()
            acc_value.append(acc.item())

            self.train_writer.add_scalar('acc', acc, self.global_step)
            self.train_writer.add_scalar('loss', loss.item(), self.global_step)

            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            timer['statistics'] += self.split_time()

            process.set_description(f'Loss: {loss.item():.4f} Acc: {acc.item():.2f}')

        self.print_log(f'learning rate: {self.lr}')
        self.print_log(f'\tMean training loss: {np.mean(loss_value):.4f}. Mean training acc: {np.mean(acc_value)*100:.2f}%.')

        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
            torch.save(weights, f"{self.arg.model_saved_name}-{epoch+1}-{int(self.global_step)}.pt")

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')

        self.model.eval()
        self.print_log(f'Eval epoch: {epoch + 1}')

        for ln in loader_name:
            loss_value = []
            score_frag = []
            label_list = []
            pred_list = []
            process = tqdm(self.data_loader[ln], ncols=80)

            for batch_idx, (data, label, index) in enumerate(process):
                label_list.append(label)
                with torch.no_grad():
                    data = data.float().to(self.device, non_blocking=True)
                    label = label.long().to(self.device, non_blocking=True)
                    output = self.model(data)
                    loss = self.loss(output, label)
                    score_frag.append(output.data.cpu().numpy())
                    loss_value.append(loss.item())

                    _, predict_label = torch.max(output.data, 1)
                    pred_list.append(predict_label.cpu().numpy())

                if wrong_file is not None or result_file is not None:
                    predict = list(predict_label.cpu().numpy())
                    true = list(label.cpu().numpy())
                    for i, x in enumerate(predict):
                        if result_file is not None:
                            f_r.write(f'{x},{true[i]}\n')
                        if x != true[i] and wrong_file is not None:
                            f_w.write(f'{index[i]},{x},{true[i]}\n')

            score = np.concatenate(score_frag)
            accuracy = self.data_loader[ln].dataset.top_k(score, 1)

            if accuracy > self.best_acc:
                self.best_acc = accuracy
                self.best_acc_epoch = epoch + 1

            print(f'Accuracy: {accuracy:.4f}, model: {self.arg.model_saved_name}')

            if self.arg.phase == 'train':
                self.val_writer.add_scalar('loss', np.mean(loss_value), self.global_step)
                self.val_writer.add_scalar('acc', accuracy, self.global_step)

            self.print_log(f'\tMean {ln} loss of {len(self.data_loader[ln])} batches: {np.mean(loss_value):.4f}')
            for k in self.arg.show_topk:
                self.print_log(f'\tTop{k}: {100 * self.data_loader[ln].dataset.top_k(score, k):.2f}%')

            if save_score:
                score_dict = dict(zip(self.data_loader[ln].dataset.sample_name, score))
                with open(f'{self.arg.work_dir}/epoch{epoch + 1}_{ln}_score.pkl', 'wb') as f:
                    pickle.dump(score_dict, f)

            # Per-class accuracy
            label_list = np.concatenate(label_list)
            pred_list = np.concatenate(pred_list)
            confusion = confusion_matrix(label_list, pred_list)
            list_diag = np.diag(confusion)
            list_raw_sum = np.sum(confusion, axis=1)
            each_acc = list_diag / (list_raw_sum + 1e-6)

            with open(f'{self.arg.work_dir}/epoch{epoch + 1}_{ln}_each_class_acc.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(each_acc)
                writer.writerows(confusion)

        if wrong_file is not None:
            f_w.close()
        if result_file is not None:
            f_r.close()

    def start(self):
        print('Starting...')
        if self.arg.phase == 'train':
            self.print_log(f'Parameters:\n{str(vars(self.arg))}\n')

            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.print_log(f'# Parameters: {count_parameters(self.model)}')

            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = (((epoch + 1) % self.arg.save_interval == 0) or
                             (epoch + 1 == self.arg.num_epoch)) and (epoch + 1) > self.arg.save_epoch

                self.train(epoch, save_model=save_model)
                self.eval(epoch, save_score=self.arg.save_score, loader_name=['test'])

            # Test best model
            weights_path = glob.glob(os.path.join(self.arg.work_dir, f'runs-{self.best_acc_epoch}*'))[0]
            weights = torch.load(weights_path, map_location=self.device)

            if isinstance(self.arg.device, list) and len(self.arg.device) > 1:
                weights = OrderedDict([[f'module.{k}', v] for k, v in weights.items()])
            self.model.load_state_dict(weights)

            wf = weights_path.replace('.pt', '_wrong.txt')
            rf = weights_path.replace('.pt', '_right.txt')
            self.arg.print_log = False
            self.eval(epoch=0, save_score=True, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.arg.print_log = True

            self.print_log(f'Best accuracy: {self.best_acc}')
            self.print_log(f'Epoch number: {self.best_acc_epoch}')

        elif self.arg.phase == 'test':
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            wf = self.arg.weights.replace('.pt', '_wrong.txt')
            rf = self.arg.weights.replace('.pt', '_right.txt')
            self.print_log(f'Model: {self.arg.model}')
            self.print_log(f'Weights: {self.arg.weights}')
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.')


if __name__ == '__main__':
    parser = get_parser()
    p = parser.parse_args()

    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print(f'WRONG ARG: {k}')
                assert k in key
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()
