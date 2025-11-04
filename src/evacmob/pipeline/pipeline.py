"""High-level orchestration for the POI language model pipeline."""

from __future__ import annotations

import logging
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
    main()
