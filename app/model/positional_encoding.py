import torch
import math

from .config import Config


class PositionalEncoding(torch.nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=config.dropout)
        positional_encoding = torch.zeros(config.seq_len, config.d_model)
        position = torch.arange(0, config.seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, config.d_model, 2) * \
            (-math.log(10_000.0) / config.d_model))
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('positional_encoding',
                             positional_encoding.unsqueeze(0))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        :param input: Input tensor of shape (batch_size, seq_len, d_model)
        :return: Output tensor with positional encoding applied.
        :rtype: Tensor
        """
        input = input + self.positional_encoding[:, :input.size(1)]
        return self.dropout(input)
