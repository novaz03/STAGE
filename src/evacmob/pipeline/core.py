"""Core orchestration logic for the evacmob pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .. import data as data_mod
from .. import preprocess as preprocess_mod
from .. import visualize as visualize_mod

from .config import PipelineArtifacts, PipelineConfig
from .features import (
    aggregate_latents_to_hex,
    cluster_latents,
    postprocess_hex_features,
    project_trajectories,
    train_autoencoder_embeddings,
)
from .llm import (
    LLMCallable,
    evaluate_poi_labels_with_llm,
    recompute_poi_embeddings_with_llm,
)
from .trips import (
    build_trajectory_features_from_segments,
    load_trip_logs,
    preprocess_trip_logs,
)


LOGGER = logging.getLogger(__name__)


def run_pipeline(
    config: PipelineConfig,
    llm_fn: LLMCallable | None = None,
) -> PipelineArtifacts:
    """Execute the streamlined pipeline, returning intermediate artefacts."""
    LOGGER.info("Starting pipeline with hex: %s and POI: %s", config.hex_path, config.poi_path)
    hex_gdf = data_mod.load_hexagon_data(config.hex_path)
    poi_gdf = data_mod.load_poi_data(config.poi_path, geometry_col=config.poi_geometry_col)

    poi_gdf = evaluate_poi_labels_with_llm(
        poi_gdf,
        llm_fn=llm_fn,
        text_col=config.poi_label_col,
        prompt=config.llm_prompt,
        batch_size=config.llm_batch_size,
    )

    if config.recompute_embeddings:
        if not config.llm_model_name:
            raise ValueError("llm_model_name must be provided when recompute_embeddings=True.")
        text_col = (
            config.poi_embedding_text_col
            if config.poi_embedding_text_col is not None
            else ("llm_label" if "llm_label" in poi_gdf.columns else config.poi_label_col)
        )
        embeddings = recompute_poi_embeddings_with_llm(
            poi_df=poi_gdf,
            text_col=text_col,
            model_name=config.llm_model_name,
            tokenizer_name=config.llm_tokenizer_name,
            device=config.device,
            batch_size=config.llm_embed_batch_size,
            max_length=config.llm_max_length,
            normalize=config.llm_normalize_embeddings,
        )
        poi_gdf = poi_gdf.assign(**{config.poi_vector_col: list(embeddings)})

    vectors = data_mod.parse_vector_column(poi_gdf[config.poi_vector_col])
    poi_latents, _ = train_autoencoder_embeddings(
        vectors,
        hidden_dim=config.autoencoder_hidden_dim,
        latent_dim=config.autoencoder_latent_dim,
        epochs=config.autoencoder_epochs,
        lr=config.autoencoder_lr,
        batch_size=config.autoencoder_batch_size,
        device=config.device,
        seed=config.random_state,
    )
    poi_gdf = poi_gdf.assign(poi_latent=list(poi_latents))

    joined = data_mod.assign_pois_to_hexagons(
        poi_gdf,
        hex_gdf,
        poi_id_col=config.poi_id_col,
        hex_id_col="hex_id",
    )

    hex_features = aggregate_latents_to_hex(joined, latent_col="poi_latent")
    hex_features = postprocess_hex_features(hex_features, strategy=config.postprocess_strategy)

    trip_segments_df = None
    trip_links_df = None
    nearest_links_df = None
    traj_source_df = None

    if config.trip_logs_path and Path(config.trip_logs_path).exists():
        raw_trip_df = load_trip_logs(config.trip_logs_path, fmt=config.trip_format)
        segments_bundle = preprocess_trip_logs(
            raw_trip_df,
            person_col=config.trip_person_col,
            join_dist_m=config.trip_join_dist_m,
            join_gap_hours=config.trip_join_gap_hours,
            hard_break_hours=config.trip_hard_break_hours,
        )
        trip_segments_df = segments_bundle.segments
        trip_links_df = segments_bundle.links
        nearest_links_df = preprocess_mod.nearest_pois_for_links(
            trip_links_df,
            poi_gdf,
            k=config.nearest_poi_k,
            n_links=config.nearest_poi_links,
            person_col=config.trip_person_col,
        )
        traj_source_df = build_trajectory_features_from_segments(
            trip_segments_df,
            person_col=config.trip_person_col,
            id_col=config.trajectory_id_col,
        )

    if config.trajectory_df_path and Path(config.trajectory_df_path).exists():
        traj_df = pd.read_parquet(config.trajectory_df_path)
    else:
        traj_df = traj_source_df

    traj_latents = None
    cluster_series = None

    if traj_df is not None and not traj_df.empty:
        feature_cols = (
            list(config.trajectory_feature_cols)
            if config.trajectory_feature_cols
            else [c for c in traj_df.columns if c != config.trajectory_id_col]
        )
        proj = project_trajectories(
            traj_df[[config.trajectory_id_col, *feature_cols]],
            feature_cols=feature_cols,
            random_state=config.random_state,
        )
        traj_latents = proj.set_index(config.trajectory_id_col)
        cluster_series = cluster_latents(
            traj_latents,
            n_clusters=config.cluster_k,
            random_state=config.random_state,
        )

    viz_path = None
    if config.states_path and config.visualization_path and config.visualization_inset_bounds:
        viz_path = visualize_mod.plot_hexgrid_with_inset(
            hex_gdf=hex_gdf,
            ses_gdf=hex_features,
            states_shapefile=config.states_path,
            inset_bounds=tuple(config.visualization_inset_bounds),
            out_path=config.visualization_path,
        )

    return PipelineArtifacts(
        hex_gdf=hex_gdf,
        poi_gdf=poi_gdf,
        poi_latents=poi_latents,
        hex_features=hex_features,
        traj_latents=traj_latents,
        cluster_labels=cluster_series,
        visualization_path=viz_path,
        trip_segments=trip_segments_df,
        trip_links=trip_links_df,
        nearest_poi_links=nearest_links_df,
    )
