"""Configuration objects for the POI language modelling pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Tuple


@dataclass
class PipelineConfig:
    """
    Parameters governing data ingestion, tokenisation, and fine-tuning.

    Defaults mirror the behaviour of ``notebooks/demo.ipynb`` so that the
    converted pipeline reproduces the original experimentation setup.
    """

    data_csv: Path = Path("../US_POI.csv")
    output_dir: Path = Path("new_pipeline_artifacts")
    tokenizer_base: str = "google/gemma-3-1b-it"
    tokenizer_dirname: str = "tokenizer"
    model_name: str = "google/gemma-3-1b-it"
    hf_token_env_var: str = "HF_TOKEN"
    block_size: int = 64
    per_device_train_batch_size: int = 256
    gradient_accumulation_steps: int = 64
    num_train_epochs: float = 1.0
    learning_rate: float = 3e-4
    dataloader_num_workers: int = 4
    fp16: bool = True
    logging_steps: int = 10
    save_steps: int = 30
    save_total_limit: int = 3
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: Sequence[str] = field(
        default_factory=lambda: ("q_proj", "k_proj", "v_proj", "o_proj")
    )
    required_special_tokens: Sequence[str] = field(
        default_factory=lambda: ("<null_val>", "[sep]")
    )
    bounding_box: Optional[Tuple[float, float, float, float]] = (
        -88.57,
        79.95,
        24.45,
        32.35,
    )
    text_columns: Sequence[str] = field(
        default_factory=lambda: ("TOP_CATEGORY", "SUB_CATEGORY", "LOCATION_NAME")
    )
    max_samples: Optional[int] = None

    def tokenizer_output_dir(self) -> Path:
        return self.output_dir / self.tokenizer_dirname

    def dataset_cache_dir(self) -> Path:
        return self.output_dir / "datasets"

    def checkpoints_dir(self) -> Path:
        return self.output_dir / "checkpoints"
