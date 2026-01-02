import torch
import os

from torch.utils.data import DataLoader

from model.config import Config
from model.stock_transformer import StockTransformer
from model.stock_dataset import StockDataset
from model.checkpointer import Checkpointer


class Trainer:
    def __init__(self, config: Config):
        self.config = config

    def train(self, model: StockTransformer, step: int = 0):
        model.train()
        checkpointer = Checkpointer(self.config)
        train_dataloader, test_dataloader = self._get_train_test_dataloaders()
        for epoch in range(self.config.epochs):
            print(f'{epoch=}')
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.lr,
                                        betas=self.config.betas)
            for inputs, targets in train_dataloader:
                optimizer.zero_grad()
                output = model(inputs).squeeze(1)
                loss = self.config.criterion(output, targets)
                if step % 100 == 0:
                    print(f'{step=}; {loss.item()=}')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                            max_norm=self.config.max_norm)
                optimizer.step()
                step += 1
            checkpointer.save_checkpoint(model, step)
        self._evaluate_model(model, test_dataloader)
    
    def _evaluate_model(self, model: StockTransformer, dataloader: DataLoader):
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for inputs, targets in dataloader:
                test_loss += self.config.criterion(model(inputs).squeeze(1),
                                                   targets).item()
        print(f'Test loss: {test_loss / len(dataloader):0.6f}')
    
    def _get_train_test_dataloaders(self):
        dataset = StockDataset(self.config)
        train_dataset_len = int(len(dataset) * 0.8)
        test_dataset_len = len(dataset) - train_dataset_len
        num_workers = os.cpu_count() or 2
        train_dataset = torch.utils.data.Subset(dataset, range(train_dataset_len))
        test_dataset = torch.utils.data.Subset(dataset, range(train_dataset_len,
                                                            train_dataset_len + test_dataset_len))
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            num_workers=num_workers
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            num_workers=num_workers
        )
        return train_dataloader, test_dataloader