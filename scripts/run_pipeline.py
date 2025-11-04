#!/usr/bin/env python
"""End-to-end POI processing pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from evacmob.pipeline import (
    AggregationConfig,
    EncodingConfig,
    MLPConfig,
    aggregate_poi_latents_to_hex,
    encode_poi_to_parquet,
    run_mlp_training,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the POI embedding pipeline")
    parser.add_argument(
        "--raw-poi-csv",
        type=Path,
        required=True,
        help="CSV containing study-area POIs with geometry",
    )
    parser.add_argument(
        "--hex-parquet", type=Path, required=True, help="Hexagon tessellation parquet"
    )
    parser.add_argument(
        "--llm-checkpoint",
        type=Path,
        required=True,
        help="Fine-tuned Hugging Face checkpoint directory",
    )
    parser.add_argument(
        "--base-model", default="google/gemma-3-1b-it", help="Base causal-LM model to load"
    )
    parser.add_argument(
        "--projection-parquet",
        type=Path,
        default=Path("POI_vec_proj_matrix.parquet"),
        help="Output parquet for encoded POIs",
    )
    parser.add_argument(
        "--mlp-checkpoint",
        type=Path,
        default=Path("models/bottleneck_mlp.pth"),
        help="Checkpoint path for the bottleneck MLP",
    )
    parser.add_argument(
        "--aggregated-parquet",
        type=Path,
        default=Path("POI_encoded_embeddings.parquet"),
        help="Output parquet for aggregated hex embeddings",
    )
    parser.add_argument(
        "--latent-column", default="z_poi", help="Column name storing bottleneck latents"
    )
    parser.add_argument(
        "--placekey-col", default="PLACEKEY", help="POI identifier column used for joins"
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for LM encoding")
    parser.add_argument(
        "--max-length", type=int, default=512, help="Max token length for LM encoding"
    )
    parser.add_argument("--skip-encode", action="store_true", help="Skip LLM encoding step")
    parser.add_argument("--skip-mlp", action="store_true", help="Skip bottleneck MLP training")
    parser.add_argument("--skip-aggregate", action="store_true", help="Skip aggregation step")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    logger = logging.getLogger("pipeline-script")

    steps = [
        ("Encoding POIs", not args.skip_encode),
        ("Training bottleneck MLP", not args.skip_mlp),
        ("Aggregating to hexagons", not args.skip_aggregate),
    ]
    total_steps = sum(flag for _, flag in steps)
    step_idx = 0

    if not args.skip_encode:
        step_idx += 1
        logger.info("Step %s/%s: %s", step_idx, total_steps, "Encoding POIs with the LLM")
        encode_poi_to_parquet(
            EncodingConfig(
                input_csv=args.raw_poi_csv,
                output_parquet=args.projection_parquet,
                checkpoint_dir=args.llm_checkpoint,
                base_model=args.base_model,
                batch_size=args.batch_size,
                max_length=args.max_length,
            )
        )
        logger.info("Finished encoding POIs → %s", args.projection_parquet)

    if not args.skip_mlp:
        step_idx += 1
        logger.info("Step %s/%s: %s", step_idx, total_steps, "Training bottleneck MLP")
        run_mlp_training(
            MLPConfig(
                input_parquet=args.projection_parquet,
                output_parquet=args.projection_parquet,
                checkpoint_path=args.mlp_checkpoint,
                vector_column="concatenated_vec",
            )
        )
        logger.info("Finished MLP training → %s", args.mlp_checkpoint)

    if not args.skip_aggregate:
        step_idx += 1
        logger.info("Step %s/%s: %s", step_idx, total_steps, "Aggregating POIs to hexagons")
        aggregate_poi_latents_to_hex(
            AggregationConfig(
                poi_geometry_csv=args.raw_poi_csv,
                poi_parquet=args.projection_parquet,
                hex_parquet=args.hex_parquet,
                output_path=args.aggregated_parquet,
                poi_id_col=args.placekey_col,
                latent_column=args.latent_column,
            )
        )
        logger.info("Finished aggregation → %s", args.aggregated_parquet)


if __name__ == "__main__":
    main()
