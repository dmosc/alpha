import torch

from .config import Config


class TimeSeriesEmbedding(torch.nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.feature_projection = torch.nn.Linear(config.input_dims,
                                                  config.d_model)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        :param input: Input tensor of shape (..., input_dims)
        :return: Output tensor with linear projection applied.
        :rtype: Tensor
        """
        assert input.shape[-1] == self.config.input_dims
        return self.feature_projection(input)
