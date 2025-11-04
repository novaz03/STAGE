"""High-level orchestration for the POI language model pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch

from .config import PipelineConfig
from .data import (
    attach_labels,
    build_datasets,
    load_poi_dataframe,
    tokenize_corpus,
)
from .modeling import load_model_with_lora
from .tokenizer import maybe_login_to_hub, prepare_tokenizer
from .training import create_trainer
from .utils import ensure_output_dirs, setup_logging

LOGGER = logging.getLogger(__name__)


def main(config: Optional[PipelineConfig] = None) -> None:
    """Run the end-to-end fine-tuning pipeline."""
    setup_logging()
    config = config or PipelineConfig()

    ensure_output_dirs(config)
    maybe_login_to_hub(config)

    df = load_poi_dataframe(config)
    dataset = build_datasets(df)
    tokenizer = prepare_tokenizer(config, dataset)
    lm_dataset = tokenize_corpus(tokenizer, dataset, config.block_size)
    lm_dataset = attach_labels(lm_dataset)

    LOGGER.info("CUDA available: %s", torch.cuda.is_available())
    model = load_model_with_lora(config, tokenizer)
    LOGGER.info("Model device: %s", next(model.parameters()).device)

    trainer = create_trainer(config, model, tokenizer, lm_dataset)
    LOGGER.info("Starting training loop")
    trainer.train()
    LOGGER.info("Training complete")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune the POI language model with LoRA.")
    parser.add_argument("--data-csv", type=Path, default=PipelineConfig.data_csv)
    parser.add_argument("--output-dir", type=Path, default=PipelineConfig.output_dir)
    parser.add_argument("--tokenizer-base", default=PipelineConfig.tokenizer_base)
    parser.add_argument("--model-name", default=PipelineConfig.model_name)
    parser.add_argument("--block-size", type=int, default=PipelineConfig.block_size)
    parser.add_argument("--epochs", type=float, default=PipelineConfig.num_train_epochs)
    parser.add_argument("--learning-rate", type=float, default=PipelineConfig.learning_rate)
    parser.add_argument(
        "--per-device-batch-size", type=int, default=PipelineConfig.per_device_train_batch_size
    )
    parser.add_argument(
        "--gradient-steps", type=int, default=PipelineConfig.gradient_accumulation_steps
    )
    parser.add_argument("--fp16", action="store_true", default=PipelineConfig.fp16)
    args = parser.parse_args()

    cfg = PipelineConfig(
        data_csv=args.data_csv,
        output_dir=args.output_dir,
        tokenizer_base=args.tokenizer_base,
        model_name=args.model_name,
        block_size=args.block_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_steps,
        fp16=args.fp16,
    )
    main(cfg)
