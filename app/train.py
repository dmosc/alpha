import torch
import os

from pathlib import Path
from torch.utils.data import DataLoader

from model.config import Config
from model.stock_transformer import StockTransformer
from model.stock_dataset import StockDataset
from model.checkpointer import Checkpointer

def main():
    _, data_dir = _init_environment()
    config = Config(data_dir)
    checkpointer = Checkpointer(config)
    model = StockTransformer(config)
    dataset = StockDataset(config)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=os.cpu_count() or 2
    )
    step = 0
    for epoch in range(config.epochs):
        print(f'{epoch=}')
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                    betas=config.betas)
        criterion = torch.nn.MSELoss()
        model.train()
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            output = model(inputs).squeeze(1)
            loss = criterion(output, targets)
            if step % config.save_every_n_steps == 0:
                print(f'{step=}; {loss.item()=}')
                checkpointer.save_checkpoint(model, step)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                        max_norm=config.max_norm)
            optimizer.step()
            step += 1


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
