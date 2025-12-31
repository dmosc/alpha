import torch

from pathlib import Path

from model.config import Config
from model.stock_transformer import StockTransformer
from model.dataloader import DataLoader


def main():
    device, data_dir = _init_environment()
    config = Config(data_dir)
    model = StockTransformer(config)
    dataloader = DataLoader(config)
    for epoch in range(config.epochs):
        print(f'{epoch=}')
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                    betas=config.betas)
        criterion = torch.nn.MSELoss()
        model.train()
        while data_paylaod := dataloader.get_next_batch():
            batch, targets = data_paylaod
            optimizer.zero_grad()
            output = model(batch).squeeze(1)
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
