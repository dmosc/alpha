from pathlib import Path


class Config:
    def __init__(self, data_dir: Path):
        self.ticker = 'SPY'
        self.epochs = 10
        self.batch_size = 16
        self.seq_len = 5
        self.input_dims = 5
        self.d_model = 150
        self.dropout = 0.1
        self.n_heads = 10
        self.d_ff = int(8 / 3) * self.d_model
        self.num_layers = 5
        self.lr = 1e-3
        self.betas = (0.9, 0.98)
        self.max_norm = 1.0
        self.data_dir = data_dir
        self.training_data_dir = self.data_dir / 'tickers'
        self.models_dir = self.data_dir / 'models'
        self.training_features = ['Open', 'High', 'Low', 'Close', 'Volume',
                                  'LogReturn']
        self.eps = 1e-8
        self.save_every_n_steps = 1_000
    
    def state_dict(self):
        return self.__dict__