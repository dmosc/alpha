import torch

from pathlib import Path


class Config:
    def __init__(self, data_dir: Path):
        self.ticker = 'SPY'
        self.epochs = 10
        self.batch_size = 32
        self.seq_len = 60
        self.training_features = ['Open', 'High', 'Low', 'Close', 'Volume',
                                  'MA_8', 'MA_20', 'MA_50', 'MA_200',
                                  'LogReturn']
        self.input_dims = len(self.training_features)
        self.target_idx = self.training_features.index('LogReturn')
        self.d_model = 300
        self.dropout = 0.1
        self.n_heads = 10
        self.d_ff = int(8 / 3) * self.d_model
        self.num_layers = 10
        self.lr = 1e-5
        self.betas = (0.9, 0.98)
        self.max_norm = 1.0
        self.data_dir = data_dir
        self.training_data_dir = self.data_dir / 'tickers'
        self.models_dir = self.data_dir / 'models'
        self.state_file = 'state.pkl'
        self.eps = 1e-8
        self.criterion = torch.nn.MSELoss()
    
    def state_dict(self):
        return self.__dict__