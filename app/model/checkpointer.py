import torch

from pathlib import Path

from .config import Config
from .stock_transformer import StockTransformer


class Checkpointer:
    def __init__(self, config: Config):
        self.config = config
    
    def save_checkpoint(self, model: StockTransformer, step: int):
        model_path = self.config.models_dir / str(step) / f'{self.config.ticker}.pkl'
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.serialization.add_safe_globals([Path])
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': self.config.__dict__,
            'step': step
        }, model_path)
        print(f'Model saved to {model_path}')