"""GRU sequence model: reads a patient's recent encounters in order, then predicts the next 30 days."""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch import nn

from workflow_ml.config import LABELS, SEED
from workflow_ml.features import feature_columns
from workflow_ml.preprocess import dense_preprocessor
from workflow_ml.split import early_stopping_split


def history_windows(table: pd.DataFrame, length: int) -> np.ndarray:
    """Row numbers of each encounter's last `length` visits (oldest first, -1 = padding).

    Relies on the table being ordered by patient then time, as `encounter_frame` guarantees.
    """
    position = table.groupby("patient_id", sort=False).cumcount().to_numpy()
    steps_back = np.arange(length - 1, -1, -1)
    rows = np.arange(len(table))[:, None] - steps_back
    return np.where(steps_back <= position[:, None], rows, -1)


class _Network(nn.Module):
    def __init__(self, n_inputs: int, hidden: int, n_labels: int, dropout: float):
        super().__init__()
        self.encode = nn.Sequential(nn.Linear(n_inputs, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_labels)
        )

    def forward(self, visits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoded = self.encode(visits) * mask.unsqueeze(-1)
        states, _ = self.gru(encoded)
        # The current visit is also passed straight to the head so the GRU only has to add history.
        return self.head(torch.cat([states[:, -1], encoded[:, -1]], dim=1))


class GRUModel:
    def __init__(self, history: int = 24, hidden: int = 160, dropout: float = 0.3,
                 epochs: int = 30, patience: int = 5, seed: int = SEED):
        self.history, self.hidden, self.dropout = history, hidden, dropout
        self.epochs, self.patience, self.seed = epochs, patience, seed
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    def _tensors(self, table: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self.preprocessor.transform(table[self.columns]).astype(np.float32)
        windows = history_windows(table, self.history)
        return torch.from_numpy(inputs).to(self.device), torch.from_numpy(windows).to(self.device)

    def _logits(self, inputs: torch.Tensor, windows: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        window = windows[rows]
        mask = (window >= 0).float()
        return self.network(inputs[window.clamp(min=0)], mask)

    def _score(self, inputs: torch.Tensor, windows: torch.Tensor, rows: np.ndarray) -> np.ndarray:
        self.network.eval()
        batches = torch.from_numpy(rows).to(self.device).split(2048)
        with torch.no_grad():
            logits = [self._logits(inputs, windows, batch) for batch in batches]
        return torch.sigmoid(torch.cat(logits)).cpu().numpy()

    def fit(self, table: pd.DataFrame, rows: np.ndarray) -> "GRUModel":
        torch.manual_seed(self.seed)
        self.columns = feature_columns(table)
        fit_rows, stop_rows = early_stopping_split(table, rows, self.seed)
        self.preprocessor = dense_preprocessor(self.columns).fit(table.iloc[fit_rows][self.columns])
        inputs, windows = self._tensors(table)
        targets = torch.from_numpy(table[list(LABELS)].to_numpy(np.float32)).to(self.device)
        y_stop = table[list(LABELS)].to_numpy()[stop_rows]

        self.network = _Network(inputs.shape[1], self.hidden, len(LABELS), self.dropout).to(self.device)
        optimiser = torch.optim.AdamW(self.network.parameters(), lr=1e-3, weight_decay=1e-3)
        loss_function = nn.BCEWithLogitsLoss()
        generator = torch.Generator().manual_seed(self.seed)
        fit_tensor = torch.from_numpy(fit_rows)

        best_score, best_state, stale = -np.inf, None, 0
        for _ in range(self.epochs):
            self.network.train()
            order = fit_tensor[torch.randperm(len(fit_tensor), generator=generator)].to(self.device)
            for batch in order.split(256):
                optimiser.zero_grad()
                loss_function(self._logits(inputs, windows, batch), targets[batch]).backward()
                optimiser.step()

            score = average_precision_score(y_stop, self._score(inputs, windows, stop_rows))
            if score > best_score:
                best_score, stale = score, 0
                best_state = {k: v.detach().clone() for k, v in self.network.state_dict().items()}
            else:
                stale += 1
                if stale >= self.patience:
                    break
        self.network.load_state_dict(best_state)
        return self

    def to(self, device: str) -> "GRUModel":
        """Move the trained network, e.g. to "cpu" before saving it for another machine."""
        self.device = torch.device(device)
        self.network.to(self.device)
        return self

    def predict_proba(self, table: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
        inputs, windows = self._tensors(table)
        return self._score(inputs, windows, np.asarray(rows))
