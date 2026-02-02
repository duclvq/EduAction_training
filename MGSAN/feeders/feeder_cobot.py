import numpy as np
import torch
from torch.utils.data import Dataset



class FeederCobot(Dataset):
            def __init__(self, data_path, split='train', window_size=89, normalization=False, debug=False):
                self.data_path = data_path
                self.split = split
                self.window_size = window_size
                self.normalization = normalization
                self.debug = debug
                self.load_data()
                if normalization:
                    self.get_mean_std()

            def set_sample_name(self):
                self.sample_name = [f"{self.split}_{i}" for i in range(len(self.data))]

            def load_data(self):
                import pickle
                with open(self.data_path, 'rb') as f:
                    obj = pickle.load(f)
                self.data = obj['pose']
                self.label = obj['label']
                if self.debug:
                    self.data = self.data[:100]
                    self.label = self.label[:100]
                self.set_sample_name()

            def get_mean_std(self):
                all_data = np.concatenate(self.data, axis=0)
                self.mean = all_data.mean(axis=0)
                self.std = all_data.std(axis=0)

            def __len__(self):
                return len(self.label)

            def __getitem__(self, index):
                data_numpy = self.data[index]
                label = self.label[index]
                # Padding or cropping to window_size
                T = data_numpy.shape[0]
                if T < self.window_size:
                    pad = np.zeros((self.window_size - T, data_numpy.shape[1], data_numpy.shape[2]))
                    data_numpy = np.concatenate([data_numpy, pad], axis=0)
                else:
                    data_numpy = data_numpy[:self.window_size]
                if self.normalization:
                    data_numpy = (data_numpy - self.mean) / (self.std + 1e-6)
                # (T, V, C) -> (C, T, V)
                data_numpy = data_numpy.transpose(2, 0, 1)
                # (C, T, V) -> (C, T, V, 1) để phù hợp với model (M=1)
                data_numpy = np.expand_dims(data_numpy, -1)
                return torch.tensor(data_numpy, dtype=torch.float32), label, index

            def top_k(self, score, top_k):
                rank = score.argsort()
                hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
                return sum(hit_top_k) * 1.0 / len(hit_top_k)
