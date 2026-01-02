from model.config import Config
from environment import Environment
from trainer import Trainer


if __name__ == '__main__':
    environment = Environment()
    config = Config(environment.data_dir)
    trainer = Trainer(config)
    trainer.train()
