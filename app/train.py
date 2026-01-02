from wakepy import keep

from model.stock_transformer import StockTransformer
from model.checkpointer import Checkpointer
from model.config import Config
from environment import Environment
from trainer import Trainer


if __name__ == '__main__':
    env = Environment()
    if checkpoint_path := env.args.checkpoint_path:
        print(f'Resuming training from {checkpoint_path}')
        model, config, step = Checkpointer.load_checkpoint(checkpoint_path)
    else:
        step = 0
        config = Config(env.data_dir)
        model = StockTransformer(config)
    with keep.presenting():
        trainer = Trainer(config)
        trainer.train(model, step)
