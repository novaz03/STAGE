"""Model loading utilities."""

from __future__ import annotations

import logging

from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase

from .config import PipelineConfig

LOGGER = logging.getLogger(__name__)


def load_model_with_lora(
    config: PipelineConfig, tokenizer: PreTrainedTokenizerBase
) -> PreTrainedModel:
    """Load the base model, resize embeddings, and wrap it with LoRA adapters."""
    LOGGER.info("Loading base model %s", config.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        attn_implementation="eager",
    )

    model.resize_token_embeddings(len(tokenizer))

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_target_modules),
    )

    model = get_peft_model(model, lora_config)
    LOGGER.info("Applied LoRA adapters to model")
    return model
