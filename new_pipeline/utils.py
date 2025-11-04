"""Shared helpers used across the pipeline modules."""

from __future__ import annotations

import logging
from pathlib import Path

from .config import PipelineConfig


def setup_logging() -> None:
    """Configure root logging for consistent pipeline diagnostics."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )


def ensure_output_dirs(config: PipelineConfig) -> None:
    """Create any directories that need to exist before training begins."""
    for path in (
        config.output_dir,
        config.tokenizer_output_dir(),
        config.dataset_cache_dir(),
        config.checkpoints_dir(),
    ):
        _mkdir(path)


def _mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
