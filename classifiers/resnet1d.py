"""Mathematically equivalent modern PyTorch implementation of the Bacteria-ID
1D ResNet architecture (Ho et al., Nature Communications 2019).

Reference: https://github.com/csho33/bacteria-ID/blob/master/resnet.py
"""

import copy
from typing import List, Optional, Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class ResidualBlock(nn.Module):
    """1D residual block with two kernel-5 convolutions and optional 1x1 projection."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=5,
            stride=stride,
            padding=2,
            dilation=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(num_features=out_channels)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=5,
            stride=1,
            padding=2,
            dilation=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(num_features=out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet1D(nn.Module):
    """Bacteria-ID 1D ResNet (exact layer, stride, and channel layout of Ho et al. 2019)."""

    def __init__(
        self,
        hidden_sizes: Sequence[int] = (100, 100, 100, 100, 100, 100),
        num_blocks: Sequence[int] = (2, 2, 2, 2, 2, 2),
        input_dim: int = 1000,
        in_channels: int = 64,
        n_classes: int = 30,
    ) -> None:
        super().__init__()
        if len(num_blocks) != len(hidden_sizes):
            raise ValueError("num_blocks and hidden_sizes must have the same length.")

        self.input_dim = input_dim
        self.in_channels = in_channels
        self.n_classes = n_classes

        self.conv1 = nn.Conv1d(
            1, self.in_channels, kernel_size=5, stride=1, padding=2, bias=False
        )
        self.bn1 = nn.BatchNorm1d(self.in_channels)

        layers: List[nn.Sequential] = []
        strides = [1] + [2] * (len(hidden_sizes) - 1)
        for idx, hidden_size in enumerate(hidden_sizes):
            layers.append(
                self._make_layer(hidden_size, num_blocks[idx], stride=strides[idx])
            )
        self.encoder = nn.Sequential(*layers)

        self.z_dim = self._get_encoding_size()
        self.linear = nn.Linear(self.z_dim, self.n_classes)

    def _make_layer(
        self, out_channels: int, num_blocks: int, stride: int = 1
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for s in strides:
            blocks.append(ResidualBlock(self.in_channels, out_channels, stride=s))
            self.in_channels = out_channels
        return nn.Sequential(*blocks)

    def _get_encoding_size(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.input_dim)
            z = self.encode(dummy)
        return int(z.size(1))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.encoder(x)
        return x.view(x.size(0), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.encode(x))


class BacteriaResNetClassifier:
    """Sklearn-like trainer/evaluator wrapper around ResNet1D."""

    def __init__(
        self,
        hidden_sizes: Sequence[int] = (100, 100, 100, 100, 100, 100),
        num_blocks: Sequence[int] = (2, 2, 2, 2, 2, 2),
        in_channels: int = 64,
        lr: float = 1e-3,
        betas: tuple = (0.5, 0.999),
        batch_size: int = 64,
        epochs: int = 30,
        patience: int = 7,
        device: Optional[str] = None,
        random_state: int = 42,
    ) -> None:
        self.hidden_sizes = tuple(hidden_sizes)
        self.num_blocks = tuple(num_blocks)
        self.in_channels = in_channels
        self.lr = lr
        self.betas = betas
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.random_state = random_state
        self.model: Optional[ResNet1D] = None

    def _set_seed(self) -> None:
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    @staticmethod
    def _to_loader(
        X: np.ndarray,
        y: Optional[np.ndarray],
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        x_tensor = torch.from_numpy(np.asarray(X, dtype=np.float32)).unsqueeze(1)
        if y is None:
            ds = TensorDataset(x_tensor)
        else:
            y_tensor = torch.from_numpy(np.asarray(y, dtype=np.int64))
            ds = TensorDataset(x_tensor, y_tensor)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> "BacteriaResNetClassifier":
        """Train ResNet1D from scratch with optional validation early stopping."""
        self._set_seed()
        input_dim = int(X_train.shape[1])
        n_classes = int(max(np.max(y_train), 0) + 1)

        self.model = ResNet1D(
            hidden_sizes=self.hidden_sizes,
            num_blocks=self.num_blocks,
            input_dim=input_dim,
            in_channels=self.in_channels,
            n_classes=n_classes,
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, betas=self.betas
        )
        criterion = nn.CrossEntropyLoss()
        train_loader = self._to_loader(X_train, y_train, self.batch_size, shuffle=True)
        val_loader = (
            self._to_loader(X_val, y_val, self.batch_size * 2, shuffle=False)
            if X_val is not None and y_val is not None
            else None
        )

        best_val_acc = -1.0
        best_state = None
        no_improve = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            for xb, yb in train_loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()

            if val_loader is not None:
                val_acc = self._eval_accuracy(val_loader)
                if verbose:
                    print(f"Epoch {epoch:02d}/{self.epochs} | Val Acc: {val_acc:.4f}")
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = copy.deepcopy(self.model.state_dict())
                    no_improve = 0
                else:
                    no_improve += 1
                    if self.patience and no_improve >= self.patience:
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def _eval_accuracy(self, loader: DataLoader) -> float:
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                preds = self.model(xb).argmax(dim=1)
                correct += int((preds == yb).sum().item())
                total += int(yb.size(0))
        return correct / max(total, 1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for spectra X of shape (N, L)."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        self.model.eval()
        loader = self._to_loader(X, None, self.batch_size * 2, shuffle=False)
        probs = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device, non_blocking=True)
                logits = self.model(xb)
                probs.append(F.softmax(logits, dim=1).cpu().numpy())
        return np.vstack(probs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for spectra X of shape (N, L)."""
        return np.argmax(self.predict_proba(X), axis=1)
