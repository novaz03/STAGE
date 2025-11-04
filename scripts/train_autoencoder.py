#!/usr/bin/env python3
"""Command-line helper to train the trajectory autoencoder."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from evacmob.pipeline import (
    AutoencoderArtifacts,
    AutoencoderDataConfig,
    AutoencoderModelConfig,
    AutoencoderTrainingConfig,
    train_autoencoder,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the trajectory autoencoder")
    parser.add_argument("--mode", choices=["real", "synthetic"], default="real")
    parser.add_argument("--real-dataset", type=Path, default=Path("GEOID_SES_point.parquet"))
    parser.add_argument("--synthetic-dataset", type=Path, default=Path("simulated_traj_points.parquet"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/trajectory_autoencoder.pth"))
    parser.add_argument("--latents", type=Path, default=Path("models/trajectory_latents.npz"))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--fixed-len", type=int, default=143)
    parser.add_argument("--start-hour", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    args = parse_args()

    data_cfg = AutoencoderDataConfig(
        data_mode=args.mode,
        real_dataset_path=args.real_dataset,
        synthetic_dataset_path=args.synthetic_dataset,
        fixed_length=args.fixed_len,
        start_hour=args.start_hour,
    )

    training_cfg = AutoencoderTrainingConfig(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        device=args.device,
        checkpoint_path=args.checkpoint,
        latent_output_path=args.latents,
    )

    model_cfg = AutoencoderModelConfig(max_len=max(args.fixed_len, 200))

    artefacts: AutoencoderArtifacts = train_autoencoder(data_cfg, model_cfg, training_cfg)

    history_path = args.checkpoint.with_suffix(".history.json")
    with history_path.open("w", encoding="utf-8") as fp:
        json.dump(artefacts.loss_history, fp, indent=2)

    logging.info("Loss history written to %s", history_path)


if __name__ == "__main__":
    main()
