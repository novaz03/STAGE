"""
Utilities for running the POI language model fine-tuning pipeline.

Key exports:
    - `PipelineConfig`: configuration dataclass for training runs.
    - `main`: high-level entrypoint that executes the fine-tuning pipeline.
    - `build_point_gdf`: helper that converts hurricane trajectory matrices into
      a point GeoDataFrame as demonstrated in the exploratory notebooks.
    - `build_concave_hull`: construct a concave hull around trajectory points.
    - `load_hexagon_grid_placeholder`: stub for upcoming hex-grid integration.
    - `MLPConfig` / `run_mlp_training`: utilities for the bottleneck MLP stage.
    - `load_finetuned_model` / `embed_texts`: helpers for embedding POI strings.
"""

from .config import PipelineConfig
from .data import build_point_gdf
from .embedding import embed_texts, load_finetuned_model, compute_null_embedding
from .aggregation import mean_vectors_by_group
from .cbg import CBGLoadConfig, load_cbgses
from .encode_poi_embeddings import EncodingConfig, encode_poi_to_parquet
from .geometry import (
    build_concave_hull,
    load_hexagon_grid_placeholder,
    generate_hex_grid_over_polygon,
    check_hex_non_overlap,
)
from .mlp import (
    MLPConfig,
    run_mlp_training,
    load_bottleneck_checkpoint,
    class_vector_from_head,
)
from .fill import fill_missing_vectors, compute_placeholder_latent
from .pipeline import main

__all__ = [
    "PipelineConfig",
    "main",
    "build_point_gdf",
    "build_concave_hull",
    "load_hexagon_grid_placeholder",
    "generate_hex_grid_over_polygon",
    "check_hex_non_overlap",
    "MLPConfig",
    "run_mlp_training",
    "load_finetuned_model",
    "compute_null_embedding",
    "embed_texts",
    "EncodingConfig",
    "encode_poi_to_parquet",
    "CBGLoadConfig",
    "load_cbgses",
    "mean_vectors_by_group",
    "load_bottleneck_checkpoint",
    "class_vector_from_head",
    "fill_missing_vectors",
    "compute_placeholder_latent",
]
