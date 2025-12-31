class Config:
    def __init__(self):
        self.epochs = 10
        self.batch_size = 32
        self.seq_len = 30
        self.input_dims = 5
        self.d_model = 150
        self.dropout = 0.1
        self.n_heads = 10
        self.d_ff = int(8 / 3) * self.d_model
        self.num_layers = 5
        self.lr = 1e-3
        self.betas = (0.9, 0.98)
        self.max_norm = 1.0