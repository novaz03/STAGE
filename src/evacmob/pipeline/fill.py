"""Helpers to fill missing embedding vectors with fallback values."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .embedding import compute_null_embedding


def fill_missing_vectors(
    series: pd.Series,
    fallback: np.ndarray,
) -> pd.Series:
    """
    Replace NaN/None entries in a vector series with a fallback vector.
    """
    fallback_list = fallback.tolist()

    def _replace(value):
        if value is None:
            return fallback_list
        if isinstance(value, float) and pd.isna(value):
            return fallback_list
        if isinstance(value, (list, tuple)):
            return value
        if isinstance(value, np.ndarray):
            return value.tolist()
        return fallback_list

    return series.apply(_replace)


def compute_placeholder_latent(
    tokenizer,
    llm_model,
    device: torch.device,
    mlp_model,
    placeholder: str = "<null_val>[sep]<null_val>",
) -> np.ndarray:
    """
    Embed the placeholder string using the LLM, then pass through the BottleneckMLP encoder.
    """
    embedding = compute_null_embedding(tokenizer, llm_model, device, placeholder)
    with torch.no_grad():
        tensor = torch.from_numpy(embedding).float().unsqueeze(0).to(device)
        latent, _ = mlp_model(tensor)
        return latent.squeeze(0).cpu().numpy()
