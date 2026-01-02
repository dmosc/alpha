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
    criterion = torch.nn.MSELoss()
    train_dataloader, test_dataloader = _get_train_test_dataloaders(config,
                                                                    dataset)
    train_model(config, checkpointer, model, train_dataloader, criterion)
    evaluate_model(model, test_dataloader, criterion)


def train_model(config: Config, checkpointer: Checkpointer,
                model: StockTransformer, dataloader: DataLoader,
                criterion: torch.nn.MSELoss):
    model.train()
    step = 0
    for epoch in range(config.epochs):
        print(f'{epoch=}')
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                    betas=config.betas)
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            output = model(inputs).squeeze(1)
            loss = criterion(output, targets)
            if step % 100 == 0:
                print(f'{step=}; {loss.item()=}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                        max_norm=config.max_norm)
            optimizer.step()
            step += 1
        checkpointer.save_checkpoint(model, step)


def evaluate_model(model: StockTransformer, dataloader: DataLoader,
                   criterion: torch.nn.MSELoss):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            test_loss += criterion(model(inputs).squeeze(1), targets).item()
    print(f'Test loss: {test_loss / len(dataloader):0.6f}')


def _get_train_test_dataloaders(config: Config, dataset: StockDataset):
    train_dataset_len = int(len(dataset) * 0.8)
    test_dataset_len = len(dataset) - train_dataset_len
    num_workers = os.cpu_count() or 2
    train_dataset = torch.utils.data.Subset(dataset, range(train_dataset_len))
    test_dataset = torch.utils.data.Subset(dataset, range(train_dataset_len,
                                                          train_dataset_len + test_dataset_len))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=num_workers
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        num_workers=num_workers
    )
    return train_dataloader, test_dataloader

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
