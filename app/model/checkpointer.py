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
        torch.save({
            'model': model.state_dict(),
            'config': self.config.__dict__,
            'step': step
        }, model_path)
        print(f'Model saved to {model_path}')
    
    @staticmethod
    def load_checkpoint(checkpoint_path: Path) -> tuple[StockTransformer, Config, int]:
        """Load model, config, and step from a checkpoint file."""
        torch.serialization.add_safe_globals([Path])
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        config_dict = checkpoint['config']
        config = Config(Path(config_dict['data_dir']))
        for key, value in config_dict.items():
            setattr(config, key, value)
        model = StockTransformer(config)
        model.load_state_dict(checkpoint['model'])
        step = checkpoint['step']
        print(f'Model loaded from {checkpoint_path}')
        return model, config, step