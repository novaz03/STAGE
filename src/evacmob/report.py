"""Reporting utilities (tables, metrics)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

try:  # optional dependency
    from tqdm.auto import tqdm as _tqdm
except ModuleNotFoundError:  # pragma: no cover
    def _tqdm(iterable, **kwargs):
        return iterable


LOGGER = logging.getLogger(__name__)


def write_text_report(text: str, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return out


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "torch is required for neural network reporting utilities."
        ) from exc
    return torch, nn


def build_bottleneck_mlp(
    input_dim: int,
    hidden_dim: int,
    latent_dim: int,
    n_classes: int,
    device: str | None = None,
):
    torch, nn = _require_torch()
    device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class BottleneckMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Linear(hidden_dim, latent_dim),
                nn.LeakyReLU(0.01, inplace=True),
            )
            self.head = nn.Linear(latent_dim, n_classes)

        def forward(self, x):
            z = self.encoder(x)
            logits = self.head(z)
            return z, logits

    model = BottleneckMLP().to(device)
    return model, device


def train_or_load_model(
    loader,
    input_dim: int,
    hidden_dim: int,
    latent_dim: int,
    checkpoint_path: str | Path,
    n_classes: int,
    class_labels: Sequence[str] | None = None,
    learning_rate: float = 1e-4,
    epochs: int = 40,
    device: str | None = None,
):
    """Port of the training loop used in ``Report_results.ipynb``."""
    torch, nn = _require_torch()
    model, device_obj = build_bottleneck_mlp(input_dim, hidden_dim, latent_dim, n_classes, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        LOGGER.info("Loading checkpoint from %s", checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location=device_obj)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return model, optimizer

    LOGGER.info("Training model for %s epochs", epochs)
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for xb, yb in _tqdm(loader, desc=f"Epoch {epoch}/{epochs}", unit="batch"):
            xb = xb.to(device_obj)
            yb = yb.to(device_obj)
            optimizer.zero_grad()
            _, logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(loader.dataset)
        LOGGER.info("Epoch %02d – loss %.4f", epoch, avg_loss)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "classes": list(class_labels) if class_labels is not None else None,
        },
        checkpoint_path,
    )
    return model, optimizer


def encode_features(model, loader, device=None):
    """Generate latent vectors using the notebook inference routine."""
    torch, _ = _require_torch()
    device_obj = torch.device(device) if device else next(model.parameters()).device
    model.eval()
    all_z = []
    with torch.no_grad():
        for xb, _ in _tqdm(loader, desc="Encoding"):
            xb = xb.to(device_obj)
            z = model.encoder(xb)
            all_z.append(z.cpu().numpy())
    return np.vstack(all_z)


def student_t_distribution(z: np.ndarray, centers: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Exactly matches the function used when estimating target distributions."""
    z_sq = np.sum((z[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    numerator = (1 + z_sq / alpha) ** (-(alpha + 1) / 2)
    q = numerator / numerator.sum(axis=1, keepdims=True)
    return q


def target_distribution(q: np.ndarray) -> np.ndarray:
    """Construct the sharpened target distribution from ``q``."""
    weight = q ** 2 / q.sum(axis=0, keepdims=True)
    return weight / weight.sum(axis=1, keepdims=True)


def collect_latents(model, loader, device=None) -> pd.DataFrame:
    """Convenience helper returning a dataframe of latent vectors per trajectory."""
    latents = encode_features(model, loader, device=device)
    ids = getattr(loader.dataset, "ids", range(latents.shape[0]))
    cols = [f"z{i}" for i in range(latents.shape[1])]
    df = pd.DataFrame(latents, columns=cols)
    df.insert(0, "traj_id", list(ids))
    return df


def plot_training_history(history: pd.DataFrame, out_path: str | Path) -> Path:
    """Generate the loss plot mirroring the notebook visual."""
    import matplotlib.pyplot as plt

    path = Path(out_path)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["epoch"], history["loss"], marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss")
    ax.grid(True, alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path
