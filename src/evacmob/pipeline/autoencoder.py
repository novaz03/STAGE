"""
Trajectory autoencoder training utilities.

This module consolidates the autoencoder logic that previously lived inside
``notebooks/cbgses-Copy1.ipynb`` so that both real and synthetic data sources
can reuse the same training pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm


LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration data classes
# ---------------------------------------------------------------------------


@dataclass
class AutoencoderDataConfig:
    """
    Configure how trajectory data is loaded prior to autoencoder training.

    Parameters
    ----------
    data_mode:
        ``"real"`` loads the real-world joined dataset,
        ``"synthetic"`` loads a synthetic/simulated dataset.
    real_dataset_path:
        Expected to contain the fully joined trajectory table with per-point
        features (e.g. ``GEOID_SES_point.parquet``).
    synthetic_dataset_path:
        Equivalent dataset but derived from synthetically generated trajectories.
    traj_id_column:
        Unique trajectory identifier column.
    time_column:
        Column enumerating the time-step index inside each trajectory.
    embedding_columns:
        Columns containing vector-like objects that should be concatenated into
        the model input (e.g. graph embeddings and POI latent vectors).
    fixed_length:
        Maximum number of time steps per trajectory; sequences longer than this
        are clipped, shorter ones are padded.
    start_hour:
        Optional start hour used for the positional encoding when explicit
        per-time ``hours`` are not available.
    """

    data_mode: Literal["real", "synthetic"] = "real"
    real_dataset_path: Path = Path("GEOID_SES_point.parquet")
    synthetic_dataset_path: Path = Path("simulated_traj_points.parquet")
    traj_id_column: str = "traj_id"
    time_column: str = "pt_idx"
    embedding_columns: Sequence[str] = ("graph_embedding", "vec_weighted_avg")
    fixed_length: int = 143
    start_hour: int = 0


@dataclass
class AutoencoderModelConfig:
    """Hyperparameters describing the Transformer autoencoder architecture."""

    d_model: int = 64
    nhead: int = 8
    num_layers: int = 4
    dropout: float = 0.1
    max_len: int = 200
    hod_harmonics: int = 1
    latent_l2_coeff: float = 1e-4
    latent_warmup_epochs: int = 50
    use_huber: bool = True
    huber_delta: float = 1.0


@dataclass
class AutoencoderTrainingConfig:
    """Training loop configuration."""

    batch_size: int = 128
    num_epochs: int = 300
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    train_split: float = 0.97
    val_split: float = 0.02
    patience: int = 50
    min_delta: float = 1e-5
    gradient_clip: float = 1.0
    device: Optional[str] = None  # e.g. "cuda", defaults to auto-detect
    checkpoint_path: Path = Path("models/trajectory_autoencoder.pth")
    latent_output_path: Path = Path("models/trajectory_latents.npz")


@dataclass
class AutoencoderArtifacts:
    """Outputs produced by ``train_autoencoder``."""

    model: "TrajTransformerAutoencoder"
    checkpoint_path: Path
    latent_path: Path
    loss_history: List[Dict[str, float]]
    latent_traj_ids: List[str]


# ---------------------------------------------------------------------------
# Datasets and collate utilities
# ---------------------------------------------------------------------------


class TrajDatasetWithTimes(Dataset):
    """Trajectory dataset storing per-trajectory feature matrices and time indices."""

    def __init__(self, feature_dict: Dict[str, np.ndarray], timeidx_dict: Dict[str, np.ndarray]):
        self.ids = list(feature_dict.keys())
        self.feature_dict = feature_dict
        self.timeidx_dict = timeidx_dict

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        traj_id = self.ids[index]
        features = torch.from_numpy(self.feature_dict[traj_id]).float()
        times = torch.from_numpy(self.timeidx_dict[traj_id]).long()
        return features, times, index, traj_id


def make_collate_fn_time(
    fixed_len: int,
    feature_dim: int,
    start_hour: int = 0,
    fill_value: float = 0.0,
):
    """
    Build a collate_fn that scatters variable-length trajectories into fixed tensors.

    Returns padded features, pad mask, observation mask, hour-of-day tensor, indices,
    and trajectory ids.
    """

    def collate_fn(batch):
        feats, times_list, idxs, traj_ids = zip(*batch)
        batch_size = len(batch)

        padded = torch.full((batch_size, fixed_len, feature_dim), fill_value, dtype=torch.float32)
        pad_mask = torch.ones(batch_size, fixed_len, dtype=torch.bool)
        obs_mask = torch.zeros(batch_size, fixed_len, feature_dim, dtype=torch.bool)

        for i, (feat, times) in enumerate(zip(feats, times_list)):
            valid = (times >= 0) & (times < fixed_len)
            if not torch.any(valid):
                continue

            t = times[valid]
            f = feat[valid]

            obs_row = torch.isfinite(f)
            f_clean = torch.nan_to_num(f, nan=0.0)

            uniq, inv = torch.unique(t, return_inverse=True)
            num_unique = uniq.numel()

            sum_feat = torch.zeros(num_unique, feature_dim, dtype=f.dtype)
            sum_feat.index_add_(0, inv, f_clean)

            cnt_feat = torch.zeros(num_unique, feature_dim, dtype=f.dtype)
            cnt_feat.index_add_(0, inv, obs_row.to(f.dtype))

            mean_feat = sum_feat / cnt_feat.clamp_min(1.0)
            obs_u = cnt_feat > 0

            padded[i, uniq] = mean_feat
            obs_mask[i, uniq] = obs_u
            pad_mask[i, uniq] = False

        hours = (torch.arange(fixed_len) + int(start_hour)) % 24
        hours = hours.unsqueeze(0).repeat(batch_size, 1).long()

        idx_tensor = torch.tensor(idxs, dtype=torch.long)
        return padded, pad_mask, obs_mask, hours, idx_tensor, list(traj_ids)

    return collate_fn


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------


class PositionalEncodingTimeOfDay(nn.Module):
    """Combined absolute + hour-of-day positional encoding."""

    def __init__(self, d_model: int, max_len: int = 500, hod_harmonics: int = 1):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

        self.hod_harmonics = int(hod_harmonics)
        self.hod_proj = nn.Linear(2 * self.hod_harmonics, d_model, bias=False)
        self.hod_scale = nn.Parameter(torch.tensor(1.0))

    @torch.no_grad()
    def _hours_from_start(self, batch: int, length: int, device: torch.device, start_hour) -> torch.Tensor:
        idx = torch.arange(length, device=device).unsqueeze(0)
        if isinstance(start_hour, int):
            start = torch.full((batch, 1), start_hour, device=device, dtype=torch.long)
        else:
            start = torch.as_tensor(start_hour, device=device).view(batch, 1).long()
        return (start + idx) % 24

    def forward(self, x: torch.Tensor, *, hours: Optional[torch.Tensor] = None, start_hour=None) -> torch.Tensor:
        batch, length, _ = x.shape
        device = x.device

        output = x + self.pe[:, :length]

        if hours is None:
            if start_hour is None:
                raise ValueError("Provide either `hours` or `start_hour` to PositionalEncodingTimeOfDay.")
            hours = self._hours_from_start(batch, length, device, start_hour)
        else:
            hours = torch.as_tensor(hours, device=device, dtype=torch.long)

        phase = 2 * np.pi * (hours.float() / 24.0)
        feats = []
        for harmonic in range(1, self.hod_harmonics + 1):
            feats.append(torch.sin(harmonic * phase))
            feats.append(torch.cos(harmonic * phase))
        hod = torch.stack(feats, dim=-1)
        hod = self.hod_proj(hod) * self.hod_scale

        return output + hod


class TrajTransformerAutoencoder(nn.Module):
    """Transformer-based sequence autoencoder with hour-of-day encoding."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_len: int = 200,
        hod_harmonics: int = 1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncodingTimeOfDay(
            d_model=d_model,
            max_len=max_len,
            hod_harmonics=hod_harmonics,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, input_dim)

    def forward(
        self,
        x: torch.Tensor,
        *,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        hours: Optional[torch.Tensor] = None,
        start_hour: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, length, _ = x.shape
        device = x.device

        src_emb = self.input_proj(x)
        src_emb = self.pos_enc(src_emb, hours=hours, start_hour=start_hour)
        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

        tgt_emb = torch.zeros(batch, length, memory.size(-1), device=device)
        tgt_emb = self.pos_enc(tgt_emb, hours=hours, start_hour=start_hour)

        decoded = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_key_padding_mask=src_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        reconstruction = self.output_layer(decoded)
        return reconstruction, memory


# ---------------------------------------------------------------------------
# Dataset preparation utilities
# ---------------------------------------------------------------------------


def _select_dataset_path(config: AutoencoderDataConfig) -> Path:
    if config.data_mode == "real":
        return config.real_dataset_path
    if config.data_mode == "synthetic":
        return config.synthetic_dataset_path
    raise ValueError(f"Unknown data_mode '{config.data_mode}'.")


def _prepare_vector(series: pd.Series) -> List[np.ndarray]:
    vectors: List[np.ndarray] = []
    dim: Optional[int] = None

    for value in series:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            if dim is None:
                vectors.append(np.empty((0,), dtype=np.float32))
            else:
                vectors.append(np.zeros(dim, dtype=np.float32))
            continue

        if isinstance(value, np.ndarray):
            arr = value.astype(np.float32)
        elif isinstance(value, (list, tuple)):
            arr = np.asarray(value, dtype=np.float32)
        elif isinstance(value, str):
            stripped = value.strip().strip("[]")
            arr = np.fromstring(stripped, sep=" ", dtype=np.float32) if stripped else np.zeros(dim or 0, dtype=np.float32)
        else:
            try:
                arr = np.asarray(value, dtype=np.float32)
            except Exception as exc:
                raise ValueError(f"Unsupported embedding type: {type(value)}") from exc

        if arr.ndim == 0:
            arr = np.atleast_1d(arr)

        if dim is None:
            dim = arr.shape[-1]
        elif arr.shape[-1] != dim:
            raise ValueError("Inconsistent embedding dimension detected")

        vectors.append(arr)

    if dim is None:
        raise ValueError("Failed to infer embedding dimension; no valid vectors found.")

    return [vec if vec.size else np.zeros(dim, dtype=np.float32) for vec in vectors]
    return vectors


def load_trajectory_features(config: AutoencoderDataConfig) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Load trajectory features and build dictionaries expected by the Dataset.
    """

    dataset_path = _select_dataset_path(config)
    LOGGER.info("Loading trajectory dataset from %s", dataset_path)
    df = pd.read_parquet(dataset_path)

    required_columns = [config.traj_id_column, config.time_column, *config.embedding_columns]
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' is missing from {dataset_path}.")

    feature_vectors: List[np.ndarray] = []
    for column in config.embedding_columns:
        vectors = _prepare_vector(df[column])
        feature_vectors.append(vectors)

    concat_arrays = [
        np.concatenate(items, axis=-1) if len(config.embedding_columns) > 1 else items[0]
        for items in zip(*feature_vectors)
    ]

    combined = df.copy()
    combined["_feature_vector"] = concat_arrays
    combined["_time_index"] = combined[config.time_column].astype(int) - 1

    grouped = combined.groupby(config.traj_id_column)
    feature_dict: Dict[str, np.ndarray] = {}
    timeidx_dict: Dict[str, np.ndarray] = {}

    for traj_id, group in grouped:
        ordered = group.sort_values(config.time_column)
        feature_array = np.stack(ordered["_feature_vector"].values)
        time_indices = np.clip(ordered["_time_index"].values, 0, config.fixed_length - 1)
        feature_dict[str(traj_id)] = feature_array
        timeidx_dict[str(traj_id)] = time_indices.astype(np.int64)

    return feature_dict, timeidx_dict


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def estimate_feature_stats(loader: DataLoader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Estimate per-feature mean/std over observed (non-padded) entries."""
    sum_, sumsq, count = None, None, None

    for batch in loader:
        x, pad_mask, obs_mask, *_ = batch
        x = x.to(device)
        pad_mask = pad_mask.to(device).bool()
        obs_mask = obs_mask.to(device).bool()

        valid = obs_mask & (~pad_mask).unsqueeze(-1)
        values = valid.float()

        if sum_ is None:
            dim = x.size(-1)
            sum_ = torch.zeros(dim, device=device)
            sumsq = torch.zeros(dim, device=device)
            count = torch.zeros(dim, device=device)

        sum_ += (x * values).sum(dim=(0, 1))
        sumsq += ((x * x) * values).sum(dim=(0, 1))
        count += values.sum(dim=(0, 1))

    mean = sum_ / count.clamp_min(1.0)
    var = (sumsq / count.clamp_min(1.0)) - mean.pow(2)
    std = var.clamp_min(1e-6).sqrt()
    return mean.detach(), std.detach()


def standardize_batch(
    x: torch.Tensor,
    obs_mask: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    pad_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scale to standardized units, masking out missing entries."""
    x_std = (x - mean.view(1, 1, -1)) / std.view(1, 1, -1)
    x_in = x_std.masked_fill(~obs_mask, 0.0)
    if pad_mask is not None:
        x_in = x_in.masked_fill(pad_mask.unsqueeze(-1), 0.0)
    return x_in, x_std


def masked_huber(
    recon: torch.Tensor,
    target: torch.Tensor,
    pad_mask: torch.Tensor,
    obs_mask: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    valid = obs_mask & (~pad_mask).unsqueeze(-1)
    diff = (recon - target)[valid]
    abs_diff = diff.abs()
    quadratic = torch.minimum(abs_diff, torch.tensor(delta, device=diff.device))
    return (0.5 * quadratic.pow(2) + delta * (abs_diff - quadratic)).mean()


def masked_mse(
    recon: torch.Tensor,
    target: torch.Tensor,
    pad_mask: torch.Tensor,
    obs_mask: torch.Tensor,
) -> torch.Tensor:
    valid = obs_mask & (~pad_mask).unsqueeze(-1)
    diff = (recon - target)[valid]
    return (diff * diff).mean()


@torch.no_grad()
def evaluate_standardized(
    model: TrajTransformerAutoencoder,
    loader: DataLoader,
    device: torch.device,
    feat_mean: torch.Tensor,
    feat_std: torch.Tensor,
) -> float:
    model.eval()
    total_sse = 0.0
    total_n = 0.0
    for x, pad_mask, obs_mask, hours, *_ in loader:
        x = x.to(device)
        pad_mask = pad_mask.to(device).bool()
        obs_mask = obs_mask.to(device).bool()

        x_in, x_std_target = standardize_batch(x, obs_mask, feat_mean, feat_std, pad_mask)
        recon, _ = model(x_in, src_key_padding_mask=pad_mask, start_hour=0)
        valid = obs_mask & (~pad_mask).unsqueeze(-1)
        diff = (recon - x_std_target)[valid]
        total_sse += float((diff * diff).sum().item())
        total_n += float(valid.sum().item())
    return total_sse / max(1.0, total_n)


@torch.no_grad()
def collect_latents(
    model: TrajTransformerAutoencoder,
    loader: DataLoader,
    device: torch.device,
    feat_mean: torch.Tensor,
    feat_std: torch.Tensor,
) -> Tuple[np.ndarray, List[str]]:
    """Collect latent vectors (memory output) for the dataset."""
    model.eval()
    latent_chunks: List[torch.Tensor] = []
    traj_ids: List[str] = []

    for x, pad_mask, obs_mask, hours, _, ids in loader:
        x = x.to(device)
        pad_mask = pad_mask.to(device).bool()
        obs_mask = obs_mask.to(device).bool()

        x_in, _ = standardize_batch(x, obs_mask, feat_mean, feat_std, pad_mask)
        _, latents = model(x_in, src_key_padding_mask=pad_mask, start_hour=0)
        latent_chunks.append(latents.cpu())
        traj_ids.extend(ids)

    latent_matrix = torch.cat(latent_chunks, dim=0).numpy()
    return latent_matrix, traj_ids


# ---------------------------------------------------------------------------
# High-level orchestration
# ---------------------------------------------------------------------------


def train_autoencoder(
    data_config: AutoencoderDataConfig,
    model_config: AutoencoderModelConfig,
    training_config: AutoencoderTrainingConfig,
) -> AutoencoderArtifacts:
    """
    Execute the full autoencoder training pipeline and persist artefacts.
    """

    feature_dict, timeidx_dict = load_trajectory_features(data_config)
    sample_key = next(iter(feature_dict))
    feature_dim = feature_dict[sample_key].shape[-1]

    dataset = TrajDatasetWithTimes(feature_dict, timeidx_dict)

    rng = np.random.default_rng(42)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)

    train_end = int(training_config.train_split * len(indices))
    val_end = train_end + int(training_config.val_split * len(indices))
    idx_train = indices[:train_end]
    idx_val = indices[train_end:val_end]
    idx_test = indices[val_end:]

    train_ds = Subset(dataset, idx_train)
    val_ds = Subset(dataset, idx_val)
    test_ds = Subset(dataset, idx_test)

    collate_fn = make_collate_fn_time(
        fixed_len=data_config.fixed_length,
        feature_dim=feature_dim,
        start_hour=data_config.start_hour,
    )

    device_str = training_config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    train_loader = DataLoader(
        train_ds,
        batch_size=training_config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=training_config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=training_config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    full_loader = DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    model = TrajTransformerAutoencoder(
        input_dim=feature_dim,
        d_model=model_config.d_model,
        nhead=model_config.nhead,
        num_layers=model_config.num_layers,
        dropout=model_config.dropout,
        max_len=max(data_config.fixed_length, model_config.max_len),
        hod_harmonics=model_config.hod_harmonics,
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_config.num_epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    feat_mean, feat_std = estimate_feature_stats(train_loader, device)
    feat_mean = feat_mean.to(device)
    feat_std = feat_std.to(device)

    best_val = float("inf")
    epochs_no_improve = 0
    loss_history: List[Dict[str, float]] = []

    training_config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Starting autoencoder training on %s (device=%s)", data_config.data_mode, device)

    for epoch in tqdm(range(training_config.num_epochs), desc="AE epochs"):
        model.train()
        total_sse = 0.0
        total_n = 0.0
        last_l2 = 0.0

        for x, pad_mask, obs_mask, hours, _, _ in train_loader:
            x = x.to(device)
            pad_mask = pad_mask.to(device).bool()
            obs_mask = obs_mask.to(device).bool()

            x_in, x_std_tgt = standardize_batch(x, obs_mask, feat_mean, feat_std, pad_mask)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                recon, latent = model(x_in, src_key_padding_mask=pad_mask, start_hour=data_config.start_hour)
                if model_config.use_huber:
                    recon_loss = masked_huber(recon, x_std_tgt, pad_mask, obs_mask, delta=model_config.huber_delta)
                else:
                    recon_loss = masked_mse(recon, x_std_tgt, pad_mask, obs_mask)

                latent_l2 = latent.pow(2).sum(dim=2).mean()
                warm = min(1.0, (epoch + 1) / max(1, model_config.latent_warmup_epochs))
                l2_loss = model_config.latent_l2_coeff * warm * latent_l2

                loss = recon_loss + l2_loss

                valid = obs_mask & (~pad_mask).unsqueeze(-1)
                diff = (recon - x_std_tgt)[valid]
                sse = (diff * diff).sum()
                n_valid = valid.sum()

            scaler.scale(loss).backward()
            if training_config.gradient_clip and training_config.gradient_clip > 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), training_config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()

            total_sse += float(sse.item())
            total_n += float(n_valid.item())
            last_l2 = float(l2_loss.detach().item())

        scheduler.step()

        train_mse = total_sse / max(1.0, total_n)
        val_mse = evaluate_standardized(model, val_loader, device, feat_mean, feat_std)
        test_mse = evaluate_standardized(model, test_loader, device, feat_mean, feat_std)

        loss_history.append(
            {
                "epoch": epoch,
                "train_mse": train_mse,
                "val_mse": val_mse,
                "test_mse": test_mse,
                "latent_l2": last_l2,
            }
        )

        LOGGER.info(
            "Epoch %03d — train_mse: %.6f  val_mse: %.6f  test_mse: %.6f  latent_l2: %.6f",
            epoch,
            train_mse,
            val_mse,
            test_mse,
            last_l2,
        )

        if val_mse + training_config.min_delta < best_val:
            best_val = val_mse
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "feat_mean": feat_mean.cpu(),
                    "feat_std": feat_std.cpu(),
                    "config": {
                        "data": data_config.__dict__,
                        "model": model_config.__dict__,
                        "training": training_config.__dict__,
                    },
                },
                training_config.checkpoint_path,
            )
            LOGGER.info("Saved best checkpoint to %s", training_config.checkpoint_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= training_config.patience:
                LOGGER.info(
                    "Early stopping after %d epochs without improvement (best_val=%.6f).",
                    training_config.patience,
                    best_val,
                )
                break

    model.load_state_dict(torch.load(training_config.checkpoint_path, map_location=device)["model_state_dict"])
    model.to(device).eval()

    latent_matrix, traj_ids = collect_latents(model, full_loader, device, feat_mean, feat_std)
    training_config.latent_output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        training_config.latent_output_path,
        latents=latent_matrix,
        traj_ids=np.array(traj_ids),
    )
    LOGGER.info("Saved latent matrix to %s (shape=%s)", training_config.latent_output_path, latent_matrix.shape)

    return AutoencoderArtifacts(
        model=model,
        checkpoint_path=training_config.checkpoint_path,
        latent_path=training_config.latent_output_path,
        loss_history=loss_history,
        latent_traj_ids=traj_ids,
    )
