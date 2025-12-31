import torch

from model.config import Config
from model.time_series_embedding import TimeSeriesEmbedding
from model.positional_encoding import PositionalEncoding


def main():
    config = Config()
    input = torch.randn((config.batch_size, config.seq_len, config.input_dims))
    time_series_embedding = TimeSeriesEmbedding(config)
    positional_encoding = PositionalEncoding(config)
    print(input.shape, time_series_embedding, positional_encoding)


if __name__ == '__main__':
    main()
