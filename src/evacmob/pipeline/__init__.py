"""
Facade exports for the evacmob pipeline package.

To keep ``python -m evacmob.pipeline.<module>`` CLI entrypoints free from
`RuntimeWarning: found in sys.modules after import` messages, heavy submodules
are imported lazily via ``__getattr__``.  Downstream code can continue to import
from ``evacmob.pipeline`` without change.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Tuple

from .config import PipelineConfig

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
    "AggregationConfig",
    "aggregate_poi_latents_to_hex",
    "load_bottleneck_checkpoint",
    "class_vector_from_head",
    "fill_missing_vectors",
    "compute_placeholder_latent",
    "AutoencoderDataConfig",
    "AutoencoderModelConfig",
    "AutoencoderTrainingConfig",
    "AutoencoderArtifacts",
    "train_autoencoder",
]


_LAZY_IMPORTS: Dict[str, Tuple[str, str]] = {
    "main": ("evacmob.pipeline.pipeline", "main"),
    "build_point_gdf": ("evacmob.pipeline.data", "build_point_gdf"),
    "build_concave_hull": ("evacmob.pipeline.geometry", "build_concave_hull"),
    "load_hexagon_grid_placeholder": ("evacmob.pipeline.geometry", "load_hexagon_grid_placeholder"),
    "generate_hex_grid_over_polygon": (
        "evacmob.pipeline.geometry",
        "generate_hex_grid_over_polygon",
    ),
    "check_hex_non_overlap": ("evacmob.pipeline.geometry", "check_hex_non_overlap"),
    "MLPConfig": ("evacmob.pipeline.mlp", "MLPConfig"),
    "run_mlp_training": ("evacmob.pipeline.mlp", "run_mlp_training"),
    "load_finetuned_model": ("evacmob.pipeline.embedding", "load_finetuned_model"),
    "compute_null_embedding": ("evacmob.pipeline.embedding", "compute_null_embedding"),
    "embed_texts": ("evacmob.pipeline.embedding", "embed_texts"),
    "EncodingConfig": ("evacmob.pipeline.encode_poi_embeddings", "EncodingConfig"),
    "encode_poi_to_parquet": ("evacmob.pipeline.encode_poi_embeddings", "encode_poi_to_parquet"),
    "CBGLoadConfig": ("evacmob.pipeline.cbg", "CBGLoadConfig"),
    "load_cbgses": ("evacmob.pipeline.cbg", "load_cbgses"),
    "mean_vectors_by_group": ("evacmob.pipeline.aggregation", "mean_vectors_by_group"),
    "AggregationConfig": ("evacmob.pipeline.aggregation", "AggregationConfig"),
    "aggregate_poi_latents_to_hex": (
        "evacmob.pipeline.aggregation",
        "aggregate_poi_latents_to_hex",
    ),
    "load_bottleneck_checkpoint": ("evacmob.pipeline.mlp", "load_bottleneck_checkpoint"),
    "class_vector_from_head": ("evacmob.pipeline.mlp", "class_vector_from_head"),
    "fill_missing_vectors": ("evacmob.pipeline.fill", "fill_missing_vectors"),
    "compute_placeholder_latent": ("evacmob.pipeline.fill", "compute_placeholder_latent"),
    "AutoencoderDataConfig": ("evacmob.pipeline.autoencoder", "AutoencoderDataConfig"),
    "AutoencoderModelConfig": ("evacmob.pipeline.autoencoder", "AutoencoderModelConfig"),
    "AutoencoderTrainingConfig": ("evacmob.pipeline.autoencoder", "AutoencoderTrainingConfig"),
    "AutoencoderArtifacts": ("evacmob.pipeline.autoencoder", "AutoencoderArtifacts"),
    "train_autoencoder": ("evacmob.pipeline.autoencoder", "train_autoencoder"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_name)
        attr = getattr(module, attr_name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module 'evacmob.pipeline' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(__all__)
