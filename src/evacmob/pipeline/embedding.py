"""Utilities for working with the fine-tuned language model embeddings."""

from __future__ import annotations

import logging
from typing import Sequence

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from peft import PeftModel

LOGGER = logging.getLogger(__name__)


def load_finetuned_model(
    checkpoint_path: str,
    base_model: str,
    device: torch.device | None = None,
):
    """
    Load the fine-tuned causal LM and tokenizer.

    Parameters mirror the notebook pipeline, where the tokenizer is saved inside
    the checkpoint directory.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Loading tokenizer from %s", checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    LOGGER.info("Loading base model %s", base_model)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        attn_implementation="eager",
    ).to(device)

    LOGGER.info("Attaching LoRA adapters from %s", checkpoint_path)
    model = PeftModel.from_pretrained(base, checkpoint_path).to(device).eval()
    return tokenizer, model, device


def embed_texts(
    texts: Sequence[str],
    tokenizer: PreTrainedTokenizerBase,
    model,
    device: torch.device,
    max_length: int = 512,
) -> torch.Tensor:
    """
    Tokenise and embed strings using the final hidden state of the fine-tuned LM.

    Returns a tensor of shape ``(len(texts), hidden_size)`` placed on CPU.
    """
    clean_texts = [
        "" if text is None else str(text).replace("[sep]", tokenizer.sep_token) for text in texts
    ]
    encoded = tokenizer(
        clean_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        return_token_type_ids=False,
    ).to(device)

    with torch.no_grad():
        outputs = model.base_model(
            input_ids=encoded.input_ids,
            attention_mask=encoded.attention_mask,
            output_hidden_states=True,
        )

    last_hidden = outputs.hidden_states[-1]
    sequence_lengths = encoded.attention_mask.sum(dim=1) - 1
    embeddings = last_hidden[torch.arange(len(clean_texts)), sequence_lengths, :]
    return embeddings.cpu()


def compute_null_embedding(
    tokenizer: PreTrainedTokenizerBase,
    model,
    device: torch.device,
    placeholder: str = "<null_val>[sep]<null_val>",
) -> np.ndarray:
    """
    Convenience wrapper to obtain the embedding of a placeholder string.
    """
    embedding = embed_texts([placeholder], tokenizer, model, device)[0]
    return embedding.numpy()
