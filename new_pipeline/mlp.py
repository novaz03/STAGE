"""Bottleneck MLP training utilities for POI embeddings."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

from .parquet_utils import read_parquet_row_groups

LOGGER = logging.getLogger(__name__)


@dataclass
class MLPConfig:
    """Configuration for bottleneck MLP training."""

    input_parquet: Path = Path("POI_vec_proj_matrix.parquet")
    output_parquet: Path = Path("POI_vec_proj_matrix.parquet")
    checkpoint_path: Path = Path("bottleneck_mlp_checkpoint.pth")

    vector_column: str = "concatenated_vec"
    concatenated_column: str = "concatenated"
    category_sep_token: str = "[sep]"

    latent_dim: int = 64
    hidden_dim: int = 256
    batch_size: int = 128
    encoding_batch_size: int = 1000
    learning_rate: float = 1e-4
    epochs: int = 50
    dataloader_workers: int = 6
    device: str = "auto"  # "cpu", "cuda", or "auto"

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


class EmbedDataset(Dataset):
    """Simple dataset wrapper around numpy feature matrix and integer labels."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.features[index], self.labels[index]


class BottleneckMLP(nn.Module):
    """Two-layer encoder with classification head."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, num_classes: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.head = nn.Linear(latent_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latents = self.encoder(x)
        logits = self.head(latents)
        return latents, logits


def parse_vector_column(column: pd.Series) -> List[np.ndarray]:
    """Parse stringified vectors like '[0.1 0.2 ...]' into numpy arrays."""
    return [
        _parse_single_vector(value)
        for value in column
    ]


def _parse_single_vector(value) -> np.ndarray:
    array = np.fromstring(str(value).strip("[]"), sep=" ").astype(np.float32)
    if array.size == 0:
        raise ValueError(f"Encountered empty vector while parsing value: {value!r}")
    return array


def prepare_projection_dataframe(
    df: pd.DataFrame,
    config: MLPConfig,
) -> pd.DataFrame:
    """
    Ensure helper columns exist on the projection dataframe.

    Adds category/subcategory splits as well as the numeric vector arrays.
    """
    sep = config.category_sep_token
    if config.concatenated_column not in df.columns:
        raise ValueError(
            f"Expected column '{config.concatenated_column}' in projection dataframe."
        )
    if config.vector_column not in df.columns:
        raise ValueError(
            f"Expected column '{config.vector_column}' in projection dataframe."
        )

    df = df.copy()
    if "geometry" in df.columns:
        df = df.drop(columns="geometry")

    split_values = df[config.concatenated_column].str.split(sep, n=2, expand=True)
    df["category"] = split_values[0]
    df["subcategory"] = split_values[1].fillna("<null_val>")
    df["label_pair"] = df["category"] + sep + df["subcategory"]
    df["vec_arr"] = parse_vector_column(df[config.vector_column])
    return df


def build_feature_matrix(
    df: pd.DataFrame,
    label_column: str = "label_pair",
) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """Construct the feature matrix and label vector from the prepared dataframe."""
    features = np.stack(df["vec_arr"].values)
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df[label_column].astype(str).values)
    return features, labels, label_encoder


def train_bottleneck_mlp(
    features: np.ndarray,
    labels: np.ndarray,
    label_encoder: LabelEncoder,
    config: MLPConfig,
) -> Tuple[BottleneckMLP, List[float]]:
    """Train the bottleneck MLP classifier."""
    device = config.resolve_device()
    dataset = EmbedDataset(features, labels)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.dataloader_workers,
    )

    model = BottleneckMLP(
        input_dim=features.shape[1],
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        num_classes=len(label_encoder.classes_),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    history: List[float] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            _, logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            batch_size = batch_features.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        epoch_loss = total_loss / max(total_samples, 1)
        history.append(epoch_loss)
        LOGGER.info("Epoch %s/%s - loss: %.6f", epoch, config.epochs, epoch_loss)

    save_checkpoint(model, optimizer, config.epochs, config.checkpoint_path, label_encoder)
    return model, history


def save_checkpoint(
    model: BottleneckMLP,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: Path,
    label_encoder: LabelEncoder,
) -> None:
    """Persist model/optimizer state for later reuse."""
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "label_classes": label_encoder.classes_.tolist(),
    }
    torch.save(checkpoint, path)
    LOGGER.info("Saved bottleneck MLP checkpoint to %s", path)


def encode_features(
    model: BottleneckMLP,
    features: np.ndarray,
    config: MLPConfig,
) -> np.ndarray:
    """Pass the feature matrix through the encoder to obtain latent codes."""
    device = config.resolve_device()
    model = model.to(device)
    model.eval()

    latents: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(features), config.encoding_batch_size):
            end = start + config.encoding_batch_size
            batch = torch.from_numpy(features[start:end]).float().to(device)
            latent_batch, _ = model(batch)
            latents.append(latent_batch.cpu().numpy())

    return np.vstack(latents)


def attach_latents_to_dataframe(
    df: pd.DataFrame,
    latents: np.ndarray,
    column_name: str = "z_poi",
) -> pd.DataFrame:
    """Append latent vectors to the dataframe as list-serialised columns."""
    df = df.copy()
    df[column_name] = [vec.astype(np.float32).tolist() for vec in latents]
    return df


def run_mlp_training(config: Optional[MLPConfig] = None) -> pd.DataFrame:
    """
    Entry point that loads the POI projection matrix, trains the MLP, and writes
    back the encoded vectors.

    Returns the encoded dataframe for further analysis.
    """
    config = config or MLPConfig()
    LOGGER.info("Loading projection matrix from %s", config.input_parquet)
    parquet = read_parquet_row_groups(config.input_parquet)
    prepared = prepare_projection_dataframe(parquet, config)
    features, labels, encoder = build_feature_matrix(prepared)
    model, _ = train_bottleneck_mlp(features, labels, encoder, config)
    latents = encode_features(model, features, config)
    encoded = attach_latents_to_dataframe(prepared, latents)

    encoded.to_parquet(config.output_parquet, index=False)
    LOGGER.info("Wrote encoded POI vectors to %s", config.output_parquet)
    return encoded


def load_bottleneck_checkpoint(
    checkpoint_path: Path,
    device: Optional[torch.device] = None,
) -> Tuple[BottleneckMLP, List[str]]:
    """
    Recreate a BottleneckMLP from checkpoint weights.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LOGGER.info("Loading bottleneck checkpoint from %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    in_dim = state_dict["encoder.0.weight"].shape[1]
    hidden_dim = state_dict["encoder.0.weight"].shape[0]
    latent_dim = state_dict["encoder.2.weight"].shape[0]
    num_classes = state_dict["head.bias"].shape[0]

    model = BottleneckMLP(in_dim, hidden_dim, latent_dim, num_classes)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    classes_raw = checkpoint.get("label_classes") or checkpoint.get("classes")
    if classes_raw is None:
        classes = [f"class_{i}" for i in range(num_classes)]
    elif isinstance(classes_raw, np.ndarray):
        classes = classes_raw.tolist()
    else:
        classes = list(classes_raw)

    return model, classes


def class_vector_from_head(
    model: BottleneckMLP,
    class_index: int = 0,
    add_bias: bool = True,
) -> np.ndarray:
    """
    Retrieve a representative vector from the classifier head weights.
    """
    with torch.no_grad():
        weight = model.head.weight[class_index].cpu()
        if add_bias:
            bias = model.head.bias[class_index].item()
            weight = weight + bias
    return weight.numpy()
