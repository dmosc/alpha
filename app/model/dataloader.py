import os
import torch
import random
import itertools

from collections import deque

from pathlib import Path
from typing import Iterator

from .config import Config


class DataLoader:
    def __init__(self, config: Config):
        self.config = config
        self.data = self._load_data(self.config.data_dir)

    def get_next_batch(self,
                       randomize: bool) -> tuple[torch.Tensor, torch.Tensor] | None:
        ...

    def _load_data(self, path: Path) -> Iterator[str]:
        ...
