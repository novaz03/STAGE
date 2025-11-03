"""LLM-related helpers for the evacmob pipeline."""

from __future__ import annotations

from typing import Callable, List, Sequence

import logging

import numpy as np
import pandas as pd

try:  # optional heavy dependencies
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

try:
    from transformers import AutoModel, AutoTokenizer
except ModuleNotFoundError:  # pragma: no cover
    AutoModel = None
    AutoTokenizer = None


LOGGER = logging.getLogger(__name__)

LLMCallable = Callable[[List[str], str | None], List[str]]


def evaluate_poi_labels_with_llm(
    poi_df: pd.DataFrame,
    llm_fn: LLMCallable | None,
    text_col: str,
    prompt: str | None = None,
    batch_size: int = 32,
    out_col: str = "llm_label",
) -> pd.DataFrame:
    """Request refined category labels from an external LLM callable."""
    if llm_fn is None:
        LOGGER.debug("No LLM callable provided; retaining %s values.", text_col)
        return poi_df.assign(**{out_col: poi_df[text_col]})

    texts = poi_df[text_col].fillna("").tolist()
    outputs: list[str] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        responses = llm_fn(batch, prompt)
        if len(responses) != len(batch):
            raise ValueError("LLM callable must return one label per input string.")
        outputs.extend(responses)
    LOGGER.info("Annotated %d POIs with LLM labels.", len(outputs))
    return poi_df.assign(**{out_col: outputs})


def load_pretrained_llm(
    model_name: str,
    tokenizer_name: str | None = None,
    device: str | None = None,
):
    """Load a Hugging Face model + tokenizer for embedding generation."""
    if AutoModel is None or AutoTokenizer is None:  # pragma: no cover
        raise ModuleNotFoundError("transformers is required to recompute LLM embeddings.")
    if torch is None:  # pragma: no cover
        raise ModuleNotFoundError("torch is required to recompute LLM embeddings.")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name)
    model = AutoModel.from_pretrained(model_name)
    device_obj = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(device_obj)
    model.eval()
    return tokenizer, model, device_obj


def embed_texts_with_llm(
    texts: Sequence[str],
    tokenizer,
    model,
    device,
    batch_size: int = 16,
    max_length: int = 256,
    normalize: bool = True,
) -> np.ndarray:
    """Embed texts by mean-pooling the model's last hidden state."""
    if torch is None:  # pragma: no cover
        raise ModuleNotFoundError("torch is required to compute LLM embeddings.")

    embeddings: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
        if not hasattr(outputs, "last_hidden_state"):  # pragma: no cover
            raise AttributeError("Model outputs must include last_hidden_state.")
        hidden = outputs.last_hidden_state
        mask = encoded.get("attention_mask")
        if mask is None:  # pragma: no cover
            raise AttributeError("Tokenizer must provide attention_mask.")
        mask = mask.unsqueeze(-1)
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        mean_emb = summed / counts
        if normalize:
            mean_emb = torch.nn.functional.normalize(mean_emb, p=2, dim=1)
        embeddings.append(mean_emb.cpu().numpy())
    return np.vstack(embeddings)


def recompute_poi_embeddings_with_llm(
    poi_df: pd.DataFrame,
    text_col: str,
    model_name: str,
    tokenizer_name: str | None,
    device: str | None,
    batch_size: int,
    max_length: int,
    normalize: bool,
) -> np.ndarray:
    """Convenience wrapper returning LLM-based embeddings for POIs."""
    tokenizer, model, device_obj = load_pretrained_llm(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        device=device,
    )
    texts = poi_df[text_col].fillna("").astype(str).tolist()
    embeddings = embed_texts_with_llm(
        texts,
        tokenizer=tokenizer,
        model=model,
        device=device_obj,
        batch_size=batch_size,
        max_length=max_length,
        normalize=normalize,
    )
    return embeddings
