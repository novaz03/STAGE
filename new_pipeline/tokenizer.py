"""Tokenizer management utilities."""

from __future__ import annotations

import logging
import os

from huggingface_hub import login as hf_login
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .config import PipelineConfig

LOGGER = logging.getLogger(__name__)


def maybe_login_to_hub(config: PipelineConfig) -> None:
    """Authenticate against the Hugging Face Hub if a token is available."""
    token = os.environ.get(config.hf_token_env_var)
    if not token:
        LOGGER.warning(
            "Env var %s not set; skipping Hugging Face login. "
            "Download may rely on cached credentials.",
            config.hf_token_env_var,
        )
        return

    LOGGER.info(
        "Authenticating with Hugging Face Hub using token env var %s",
        config.hf_token_env_var,
    )
    hf_login(token)


def prepare_tokenizer(
    config: PipelineConfig, dataset
) -> PreTrainedTokenizerBase:
    """
    Load the base tokenizer, ensure special tokens exist, and persist it.
    """
    LOGGER.info("Loading base tokenizer %s", config.tokenizer_base)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_base,
            use_fast=True,
        )
    except Exception:
        LOGGER.warning(
            "Falling back to slow tokenizer implementation for %s",
            config.tokenizer_base,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_base,
            use_fast=False,
        )

    added_count = tokenizer.add_special_tokens(
        {"additional_special_tokens": list(config.required_special_tokens)}
    )
    LOGGER.info("Added %s special tokens", added_count)

    tokenizer.save_pretrained(config.tokenizer_output_dir())
    LOGGER.info("Saved tokenizer to %s", config.tokenizer_output_dir())

    sample_text = dataset[0]["text"]
    LOGGER.debug(
        "Tokenizer sample tokens: %s",
        tokenizer.tokenize(sample_text)[:20],
    )
    return tokenizer
