import torch

from pathlib import Path

from model.config import Config
from trainer import Trainer

def main():
    _, data_dir = _init_environment()
    config = Config(data_dir)
    trainer = Trainer(config)
    trainer.train()


def _init_environment() -> tuple[torch.device, Path]:
    torch.manual_seed(42)
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print('Using MPS.')
    else:
        device = torch.device('cpu')
        print('Using CPU.')
    data_dir = Path(__file__).parent / 'data'
    return device, data_dir


if __name__ == '__main__':
    main()
