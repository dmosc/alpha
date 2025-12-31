import torch
import pandas as pd

from torch.utils.data import Dataset

from .config import Config


class StockDataset(Dataset):
    def __init__(self, config: Config):
        self.config = config
        paths = sorted(config.training_data_dir.glob(f"{config.ticker}_*.csv"))
        dataframe = pd.concat([pd.read_csv(p) for p in paths])
        # 2. Extract features and scale (Z-score normalization)
        data = dataframe[config.training_features].values.astype('float32')
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)
        self.normalized_data = (data - self.mean) / (self.std + config.eps)
        self.data_tensor = torch.tensor(self.normalized_data)
        self.close_idx = config.training_features.index('Close')

    def __len__(self):
        # Total windows available
        return len(self.data_tensor) - self.config.seq_len

    def __getitem__(self, idx):
        # Returns one (seq_len, features) input and one target
        x = self.data_tensor[idx : idx + self.config.seq_len]
        y = self.data_tensor[idx + self.config.seq_len, self.close_idx]
        return x, y

    def denormalize_close(self, val):
        return (val * self.std[self.close_idx]) + self.mean[self.close_idx]