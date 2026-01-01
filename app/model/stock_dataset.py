import torch
import pandas as pd

from torch.utils.data import Dataset

from .config import Config


class StockDataset(Dataset):
    def __init__(self, config: Config):
        self.config = config
        paths = sorted(config.training_data_dir.glob(f"{config.ticker}_*.csv"))
        dataframe = pd.concat([pd.read_csv(p) for p in paths])
        self.data = torch.tensor(dataframe[config.training_features].values,
                                 dtype=torch.float)
        self.targe_idx = config.training_features.index('Close')

    def __len__(self):
        return len(self.data) - self.config.seq_len

    def __getitem__(self, idx):
        input = self.data[idx : idx + self.config.seq_len]
        target = self.data[idx + self.config.seq_len, self.targe_idx]
        return input, target
