"""Configuration dataclasses for the evacmob pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import geopandas as gpd
import numpy as np
import pandas as pd


@dataclass
class PipelineConfig:
    hex_path: str | Path
    poi_path: str | Path
    states_path: str | Path | None = None
    poi_geometry_col: str = "geometry"
    poi_label_col: str = "label_pair"
    poi_vector_col: str = "z"
    poi_id_col: str = "poi_id"
    autoencoder_hidden_dim: int = 256
    autoencoder_latent_dim: int = 64
    autoencoder_epochs: int = 20
    autoencoder_lr: float = 1e-3
    autoencoder_batch_size: int = 256
    cluster_k: int = 5
    recompute_embeddings: bool = False
    poi_embedding_text_col: str | None = None
    trip_logs_path: str | Path | None = None
    trip_format: str = "auto"
    trip_person_col: str = "participantId"
    trip_join_dist_m: float = 100.0
    trip_join_gap_hours: float = 4.0
    trip_hard_break_hours: float = 8.0
    nearest_poi_k: int = 10
    nearest_poi_links: int | None = 15
    trajectory_df_path: str | Path | None = None
    trajectory_id_col: str = "traj_id"
    trajectory_feature_cols: Sequence[str] | None = None
    visualization_path: str | Path | None = None
    visualization_inset_bounds: Sequence[float] | None = None
    device: str | None = None
    llm_prompt: str | None = None
    llm_batch_size: int = 32
    llm_model_name: str | None = None
    llm_tokenizer_name: str | None = None
    llm_max_length: int = 256
    llm_embed_batch_size: int = 16
    llm_normalize_embeddings: bool = True
    random_state: int = 42
    postprocess_strategy: str = "mean"


@dataclass
class PipelineArtifacts:
    hex_gdf: gpd.GeoDataFrame
    poi_gdf: pd.DataFrame | gpd.GeoDataFrame
    poi_latents: np.ndarray
    hex_features: gpd.GeoDataFrame
    traj_latents: pd.DataFrame | None
    cluster_labels: pd.Series | None
    visualization_path: Path | None
    trip_segments: gpd.GeoDataFrame | None = None
    trip_links: gpd.GeoDataFrame | None = None
    nearest_poi_links: pd.DataFrame | None = None
