"""Structured pipeline utilities for evacmob."""

from .config import PipelineArtifacts, PipelineConfig
from .core import run_pipeline
from .features import (
    aggregate_latents_to_hex,
    cluster_latents,
    postprocess_hex_features,
    project_trajectories,
    train_autoencoder_embeddings,
)
from .llm import (
    LLMCallable,
    embed_texts_with_llm,
    evaluate_poi_labels_with_llm,
    load_pretrained_llm,
    recompute_poi_embeddings_with_llm,
)
from .trips import (
    build_trajectory_features_from_segments,
    load_trip_logs,
    preprocess_trip_logs,
)

__all__ = [
    "PipelineConfig",
    "PipelineArtifacts",
    "run_pipeline",
    "LLMCallable",
    "evaluate_poi_labels_with_llm",
    "load_pretrained_llm",
    "embed_texts_with_llm",
    "recompute_poi_embeddings_with_llm",
    "train_autoencoder_embeddings",
    "aggregate_latents_to_hex",
    "postprocess_hex_features",
    "project_trajectories",
    "cluster_latents",
    "load_trip_logs",
    "preprocess_trip_logs",
    "build_trajectory_features_from_segments",
]
