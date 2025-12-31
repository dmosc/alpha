import torch

from .config import Config
from .time_series_embedding import TimeSeriesEmbedding
from .positional_encoding import PositionalEncoding


class StockTransformer(torch.nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.time_series_embedding = TimeSeriesEmbedding(config)
        self.positional_encoding = PositionalEncoding(config)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=config.d_model,
                                                   nhead=config.n_heads,
                                                   dim_feedforward=config.d_ff,
                                                   dropout=config.dropout,
                                                   batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer,
                                                               num_layers=config.num_layers)
        # Maps the final state to a single output that represents the price
        # prediction.
        self.regressor = torch.nn.Linear(config.d_model, 1)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = self.time_series_embedding(input)
        input = self.positional_encoding(input)
        input = self.transformer_encoder(input)
        # For every item in batch grab the last seq_len vector along all its
        # input_dims since that represents the next day price prediction.
        last_step = input[:, -1, :]
        return self.regressor(last_step)