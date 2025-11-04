"""Utility script to encode POI text fields with the fine-tuned LLM."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import logging

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

LOGGER = logging.getLogger(__name__)


@dataclass
class EncodingConfig:
    """Runtime configuration for the POI embedding exporter."""

    input_csv: Path
    output_parquet: Path
    checkpoint_dir: Path
    base_model: str = "google/gemma-3-1b-it"
    batch_size: int = 256
    max_length: int = 512


def _load_checkpoint(config: EncodingConfig):
    """Load tokenizer and causal LM (optionally with LoRA adapters)."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_dir = Path(config.checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"LoRA checkpoint directory '{checkpoint_dir}' not found. Run the fine-tuning step first."
        )

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

    base = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        attn_implementation="eager",
    ).to(device)

    adapter_path = checkpoint_dir / "adapter_config.json"
    if not adapter_path.exists():
        raise FileNotFoundError(
            f"adapter_config.json not found in '{checkpoint_dir}'. Run the fine-tuning step to produce LoRA adapters before encoding."
        )

    LOGGER.info("Applying LoRA adapters from %s", checkpoint_dir)
    model = (
        PeftModel.from_pretrained(  # type: ignore[arg-type]
            base,
            checkpoint_dir,
            is_trainable=False,
            strict=False,
        )
        .to(device)
        .eval()
    )

    return tokenizer, model, device


def _ensure_concatenated(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the dataframe exposes the concatenated text column."""
    if "concatenated" in df.columns:
        return df

    required = ("TOP_CATEGORY", "SUB_CATEGORY", "LOCATION_NAME")
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns required to build 'concatenated': {', '.join(missing)}")

    df = df.copy()
    df["concatenated"] = (
        df["TOP_CATEGORY"].fillna("<null_val>").astype(str)
        + "[sep]"
        + df["SUB_CATEGORY"].fillna("<null_val>").astype(str)
        + "[sep]"
        + df["LOCATION_NAME"].fillna("<null_val>").astype(str)
    )
    return df


def _embed_texts(
    texts: Sequence[str],
    batch_size: int,
    tokenizer,
    model,
    device: torch.device,
    max_length: int,
) -> np.ndarray:
    """Batch texts through the model and collect final token embeddings."""
    vectors: List[np.ndarray] = []

    iterator = range(0, len(texts), batch_size)
    for start in tqdm(iterator, desc="Encoding POIs", unit="batch"):
        batch = texts[start : start + batch_size]
        if not batch:
            continue

        encoded = tokenizer(
            batch,
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
                return_dict=True,
            )

        last_hidden = outputs.hidden_states[-1]  # (B, L, H)
        seq_lens = encoded.attention_mask.sum(dim=1) - 1
        embeddings = last_hidden[
            torch.arange(last_hidden.size(0), device=seq_lens.device), seq_lens, :
        ]
        vectors.append(embeddings.cpu().numpy())

    return np.vstack(vectors)


def _serialize_vectors(vectors: np.ndarray) -> List[str]:
    """Convert vectors to string form for compatibility with existing pipelines."""
    return [
        "[" + " ".join(f"{float(value):.6f}" for value in row.tolist()) + "]" for row in vectors
    ]


def encode_poi_to_parquet(config: EncodingConfig) -> pd.DataFrame:
    """
    Embed POI text fields using the fine-tuned LLM and write a projection parquet.

    Returns the dataframe that was written for further use in memory.
    """
    tokenizer, model, device = _load_checkpoint(config)
    df = pd.read_csv(config.input_csv)
    df = _ensure_concatenated(df)

    texts = df["concatenated"].astype(str).tolist()
    vectors = _embed_texts(
        texts,
        batch_size=config.batch_size,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=config.max_length,
    )
    df["concatenated_vec"] = _serialize_vectors(vectors)

    columns = [
        col
        for col in (
            "PLACEKEY",
            "LONGITUDE",
            "LATITUDE",
            "REGION",
            "LOCATION_NAME",
            "concatenated",
            "concatenated_vec",
        )
        if col in df.columns
    ]
    out_df = df[columns].copy()
    out_df.to_parquet(config.output_parquet, index=False)
    return out_df


def _parse_args() -> EncodingConfig:
    parser = argparse.ArgumentParser(
        description="Encode POI strings using the fine-tuned LLM and export a parquet."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="CSV file containing POIs with concatenated text columns.",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        required=True,
        help="Destination parquet path for the encoded POI matrix.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Directory containing the fine-tuned checkpoint (tokenizer + LoRA weights).",
    )
    parser.add_argument(
        "--base-model",
        default="google/gemma-3-1b-it",
        help="Base model identifier to load before applying LoRA adapters.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for embedding the POI text.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum token count to feed into the model.",
    )
    args = parser.parse_args()
    return EncodingConfig(
        input_csv=args.input_csv,
        output_parquet=args.output_parquet,
        checkpoint_dir=args.checkpoint_dir,
        base_model=args.base_model,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )


def main() -> None:
    config = _parse_args()
    encode_poi_to_parquet(config)


if __name__ == "__main__":
    main()
