import torch
import argparse

from pathlib import Path


class Environment:
    def __init__(self):
        self.args = self._init_args()
        self.device = self._init_device()
        self.data_dir = Path(__file__).parent / 'data'

    def _init_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument('--checkpoint_path', type=str, default=None,
                            help='Path to a model checkpoint to resume training from.')
        args = parser.parse_args()
        return args
    
    def _init_device(self) -> torch.device:
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            print('Using MPS.')
        else:
            device = torch.device('cpu')
            print('Using CPU.')
        return device