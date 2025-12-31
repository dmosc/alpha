import torch
import csv

from typing import Iterator

from .config import Config


class DataLoader:
    def __init__(self, config: Config):
        self.config = config
        self.data = self._load_data()

    def get_next_batch(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return the next training batch as (inputs, targets).

        Inputs shape: (batch_size, seq_len, input_dims)
        Targets shape: (batch_size,) containing the next-day Close value for
        each example. When the dataset is exhausted the internal iterator is
        reset and None is returned to signal the caller.
        """
        close_idx = self.config.training_features.index('Close')
        inputs: list[list[list[float]]] = []
        targets: list[float] = []
        idx = 0
        while idx < self.config.batch_size:
            sequence: list[list[float]] = []
            for _ in range(self.config.seq_len + 1):
                row = next(self.data, None)
                if row is None:
                    self.data = self._load_data()
                    print('Finished processing all data; resetting pointer to start.')
                    return None
                vals = [float(val) for val in row]
                sequence.append(vals)
            inputs.append(sequence[:self.config.seq_len])
            targets.append(sequence[-1][close_idx])
            idx += 1
        return torch.tensor(inputs, dtype=torch.float), torch.tensor(targets,
                                                                     dtype=torch.float)


    def _load_data(self) -> Iterator[list[str]]:
        """Yield rows from all CSV files for the configured ticker.

        Builds rows from CSV files named `{ticker}_YYYYMMDD_YYYYMMDD.csv` with
        the specific columns declared in the model config.
        """
        if not self.config.training_data_dir.exists():
            return iter(())
        paths = sorted(self.config.training_data_dir.glob(f"{self.config.ticker}_*.csv"))
        for path in paths:
            with path.open() as file:
                reader = csv.reader(file)
                headers = next(reader, None)
                assert headers, 'Headers are required in the CSV.'
                assert headers == self.config.training_features, f'CSV columns must match {self.config.training_features=}'
                for row in reader:
                    yield row
