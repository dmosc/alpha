import torch

from pathlib import Path

from model.config import Config
from model.stock_transformer import StockTransformer


def main():
    device, data_dir = _init_environment()
    config = Config(data_dir)
    model = StockTransformer(config).to(device)
    for epoch in range(config.epochs):
        print(f'{epoch=}')
        input = torch.randn((config.batch_size, config.seq_len,
                            config.input_dims)).to(device)
        # We want to extract the last feature of the last sequence vector which
        # contains the closing price of every day and shift that by one position to
        # create the targets for every training example.
        #
        # Values for each input dimension [Open, High, Low, Volume, Close].
        target = input[:, -1, -1].roll(shifts=-1, dims=-1).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                    betas=config.betas)
        criterion = torch.nn.MSELoss()
        model.train()
        optimizer.zero_grad()
        output = model(input).squeeze(1)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       max_norm=config.max_norm)
        optimizer.step()
        print(f'{loss=}')


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
