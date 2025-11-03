"""Feature-engineering helpers for the evacmob pipeline."""

from __future__ import annotations

from typing import Iterable

import logging

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:  # optional dependency
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    nn = None


LOGGER = logging.getLogger(__name__)


def train_autoencoder_embeddings(
    vectors: np.ndarray,
    hidden_dim: int,
    latent_dim: int,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, object]:
    """Train a light autoencoder to compress POI vectors."""
    if torch is None or nn is None:  # pragma: no cover
        raise ModuleNotFoundError("torch is required to train the autoencoder.")

    device_obj = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    input_dim = vectors.shape[1]
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(vectors.astype(np.float32))
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed or 0),
    )

    class Autoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
            )

        def forward(self, x):
            z = self.encoder(x)
            recon = self.decoder(z)
            return z, recon

    model = Autoencoder().to(device_obj)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for (xb,) in loader:
            xb = xb.to(device_obj)
            optimizer.zero_grad()
            z, recon = model(xb)
            loss = criterion(recon, xb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        LOGGER.debug(
            "Autoencoder epoch %d/%d – loss %.4f",
            epoch,
            epochs,
            total_loss / len(dataset),
        )

    model.eval()
    with torch.no_grad():
        latents = []
        for (xb,) in loader:
            xb = xb.to(device_obj)
            z, _ = model(xb)
            latents.append(z.cpu().numpy())
    latent_array = np.concatenate(latents, axis=0)
    return latent_array, model


def aggregate_latents_to_hex(
    joined_poi_hex: gpd.GeoDataFrame,
    latent_col: str,
    hex_id_col: str = "hex_id",
) -> gpd.GeoDataFrame:
    """Aggregate POI latents per hexagon (mean)."""

    def _stack_mean(arrays: Iterable[np.ndarray]) -> np.ndarray:
        stacked = np.stack([np.asarray(a) for a in arrays])
        return stacked.mean(axis=0)

    agg = joined_poi_hex.groupby(hex_id_col)[latent_col].agg(_stack_mean)
    features = (
        joined_poi_hex[["hex_id", "geometry"]]
        .drop_duplicates(subset="hex_id")
        .merge(agg.rename("poi_latent"), on="hex_id", how="left")
    )
    return gpd.GeoDataFrame(features, geometry="geometry", crs=joined_poi_hex.crs)


def postprocess_hex_features(
    hex_features: gpd.GeoDataFrame,
    strategy: str = "mean",
) -> gpd.GeoDataFrame:
    """Fill missing latent vectors using the desired strategy."""
    result = hex_features.copy()
    if "poi_latent" not in result.columns:
        return result
    mask = result["poi_latent"].isna()
    if not mask.any():
        return result

    non_null = result.loc[~mask, "poi_latent"].dropna()
    if non_null.empty:
        raise ValueError("Cannot postprocess hex features – all latent vectors are missing.")
    exemplar = np.asarray(non_null.iloc[0])

    if strategy == "zero":
        fill_val = np.zeros_like(exemplar)
    elif strategy == "mean":
        fill_val = np.stack(non_null.apply(np.asarray)).mean(axis=0)
    else:
        raise ValueError(f"Unknown postprocess strategy '{strategy}'.")

    result.loc[mask, "poi_latent"] = [fill_val for _ in range(mask.sum())]
    return result


def project_trajectories(
    traj_df: pd.DataFrame,
    feature_cols: Iterable[str],
    latent_dim: int = 8,
    random_state: int = 42,
) -> pd.DataFrame:
    """Project trajectory-level features into a low-dimensional latent space."""
    scaler = StandardScaler()
    feats = scaler.fit_transform(traj_df[list(feature_cols)])
    pca = PCA(n_components=latent_dim, random_state=random_state)
    latent = pca.fit_transform(feats)
    cols = [f"traj_latent_{i}" for i in range(latent.shape[1])]
    result = traj_df[[traj_df.columns[0]]].copy()
    for idx, col in enumerate(cols):
        result[col] = latent[:, idx]
    return result


def cluster_latents(
    latents: pd.DataFrame,
    n_clusters: int,
    random_state: int = 42,
) -> pd.Series:
    """Cluster latent representations using K-Means."""
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
    labels = kmeans.fit_predict(latents.to_numpy())
    return pd.Series(labels, index=latents.index, name="cluster")
