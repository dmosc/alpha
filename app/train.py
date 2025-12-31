import torch
import os

from pathlib import Path
from torch.utils.data import DataLoader

from model.config import Config
from model.stock_transformer import StockTransformer
from model.stock_dataset import StockDataset


def main():
    _, data_dir = _init_environment()
    config = Config(data_dir)
    model = StockTransformer(config)
    dataset = StockDataset(config)
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=os.cpu_count() or 2
    )
    for epoch in range(config.epochs):
        print(f'{epoch=}')
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                    betas=config.betas)
        criterion = torch.nn.MSELoss()
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            output = model(inputs).squeeze(1)
            loss = criterion(output, targets)
            print(f'{loss.item()=}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                        max_norm=config.max_norm)
            optimizer.step()


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
