import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset

from .config import Config


class StockDataset(Dataset):
    def __init__(self, config: Config):
        self.config = config
        paths = sorted(config.training_data_dir.glob(f"{config.ticker}_*.csv"))
        dataframe = pd.concat([pd.read_csv(p) for p in paths])
        self.data = self._apply_transformations(dataframe)

    def __len__(self):
        return len(self.data) - self.config.seq_len

    def __getitem__(self, idx: int):
        input = self.data[idx : idx + self.config.seq_len]
        target = self.data[idx + self.config.seq_len, self.config.target_idx]
        return input, target

    def _apply_transformations(self, dataframe: pd.DataFrame) -> torch.Tensor:
        dataframe['LogReturn'] = np.log(
            dataframe['Close'] / dataframe['Close'].shift(1))
        dataframe.loc[0, 'LogReturn'] = 0
        return torch.tensor(dataframe[self.config.training_features].values,
                                 dtype=torch.float)
