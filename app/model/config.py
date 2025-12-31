class Config:
    def __init__(self):
        self.batch_size = 32
        self.seq_len = 30
        self.input_dims = 6
        self.d_model = 120
        self.dropout = 0.1
        self.n_heads = 6
        self.d_ff = int(8 / 3) * self.d_model
        self.num_layers = 5
