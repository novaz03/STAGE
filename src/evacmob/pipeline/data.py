"""Data loading and preprocessing utilities."""

from __future__ import annotations

import ast
import logging
from itertools import chain
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import geopandas as gpd
from datasets import Dataset
from shapely.geometry import Point

from .config import PipelineConfig

LOGGER = logging.getLogger(__name__)


def load_poi_dataframe(config: PipelineConfig) -> pd.DataFrame:
    """
    Read the raw POI CSV, fill missing values, and concatenate text columns.

    Returns a DataFrame with a ``concatenated`` column ready for tokenisation.
    """
    LOGGER.info("Loading POI data from %s", config.data_csv)
    df = pd.read_csv(config.data_csv)

    for required_column in ("LONGITUDE", "LATITUDE"):
        if required_column not in df.columns:
            raise ValueError(f"Expected column '{required_column}' in POI CSV.")

    for column in config.text_columns:
        if column not in df.columns:
            raise ValueError(f"Expected column '{column}' in POI CSV.")
        df[column] = df[column].fillna("<null_val>")

    def concat_columns(row: pd.Series) -> str:
        return "[sep]".join(str(row[col]) for col in config.text_columns)

    df["concatenated"] = df.apply(concat_columns, axis=1)

    if config.bounding_box:
        min_lon, max_lon, min_lat, max_lat = config.bounding_box
        LOGGER.info(
            "Filtering POIs to bounding box lon[%s, %s], lat[%s, %s]",
            min_lon,
            max_lon,
            min_lat,
            max_lat,
        )
        df = df[
            (df["LONGITUDE"] >= min_lon)
            & (df["LONGITUDE"] <= max_lon)
            & (df["LATITUDE"] >= min_lat)
            & (df["LATITUDE"] <= max_lat)
        ]

    if config.max_samples:
        LOGGER.info("Subsampling to %s rows for quick experiments", config.max_samples)
        df = df.sample(config.max_samples, random_state=42)

    LOGGER.info("Prepared POI dataframe with %s rows", len(df))
    return df.reset_index(drop=True)


def build_datasets(df: pd.DataFrame) -> Dataset:
    """Construct a Hugging Face Dataset from the concatenated POI text."""
    LOGGER.info("Building tokenizer dataset from concatenated column")
    dataset = Dataset.from_dict({"text": df["concatenated"].tolist()})
    if dataset.num_rows == 0:
        raise RuntimeError("No rows available after preprocessing; aborting.")
    return dataset


def tokenize_corpus(
    tokenizer,
    dataset: Dataset,
    block_size: int,
) -> Dataset:
    """Tokenise and chunk the dataset into fixed-length blocks."""
    LOGGER.info("Tokenising corpus with block size %s", block_size)

    def tokenize_fn(examples: dict) -> dict:
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            return_attention_mask=True,
        )

    tokenised = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    def group_texts(examples: dict) -> dict:
        input_ids: List[List[int]] = examples["input_ids"]
        concatenated: List[int] = list(chain.from_iterable(input_ids))
        total_length = (len(concatenated) // block_size) * block_size
        if total_length == 0:
            return {"input_ids": [], "attention_mask": []}

        concatenated = concatenated[:total_length]
        chunks = [concatenated[i : i + block_size] for i in range(0, total_length, block_size)]
        masks = [[1] * block_size for _ in chunks]
        return {"input_ids": chunks, "attention_mask": masks}

    lm_dataset = tokenised.map(
        group_texts,
        batched=True,
        batch_size=1024,
        remove_columns=tokenised.column_names,
    )

    if len(lm_dataset) == 0:
        raise RuntimeError(
            "Language modelling dataset is empty; adjust block_size or review input data."
        )

    LOGGER.info("LM dataset prepared with %s blocks", len(lm_dataset))
    return lm_dataset


def attach_labels(dataset: Dataset) -> Dataset:
    """Mirror ``input_ids`` into ``labels``; required for causal LM training."""
    return dataset.map(
        lambda batch: {"labels": batch["input_ids"]},
        batched=True,
    )


def build_point_gdf(csv_path: Path) -> gpd.GeoDataFrame:
    """
    Convert a wide hurricane trajectory matrix into a GeoDataFrame of points.

    This mirrors the logic in ``notebooks/Cleaned_pipeline.ipynb`` where the
    stacked long-form dataframe is expanded into geometry records.
    """
    LOGGER.info("Loading trajectory matrix from %s", csv_path)
    df = pd.read_csv(csv_path)

    if "traj_id" not in df.columns:
        if "Unnamed: 0" in df.columns:
            df = df.rename(columns={"Unnamed: 0": "traj_id"})
        else:
            raise ValueError("Expected 'traj_id' column or an unnamed index column.")

    df = df.set_index("traj_id")
    df_long = (
        df.stack(dropna=False).rename("coords").reset_index().rename(columns={"level_1": "pt_idx"})
    )

    df_long["coords"] = df_long["coords"].apply(_parse_coord_string)

    bad_coords = df_long["coords"].apply(lambda x: not _is_valid_pair(x))
    if bad_coords.any():
        examples = df_long.loc[bad_coords, ["traj_id", "pt_idx", "coords"]].head()
        raise ValueError(f"Failed to parse some coordinate strings; first offences:\n{examples}")

    df_long[["latitude", "longitude"]] = pd.DataFrame(
        df_long["coords"].tolist(), index=df_long.index, columns=["latitude", "longitude"]
    )

    df_long["pt_idx"] = _coerce_int(df_long["pt_idx"])
    df_long = df_long.drop(columns="coords")
    df_long = df_long.dropna(subset=["latitude", "longitude"])

    df_long["geometry"] = df_long.apply(
        lambda r: Point(r.longitude, r.latitude),
        axis=1,
    )
    gdf = gpd.GeoDataFrame(
        df_long[["traj_id", "pt_idx", "latitude", "longitude", "geometry"]],
        geometry="geometry",
        crs="EPSG:4326",
    )
    LOGGER.info("Constructed GeoDataFrame with %s trajectory points", len(gdf))
    return gdf


def assign_pois_to_hexagons(
    poi_gdf: gpd.GeoDataFrame,
    hex_gdf: gpd.GeoDataFrame,
    poi_id_col: str = "PLACEKEY",
    hex_id_col: str = "hex_id",
    projected_crs: int | str = 5070,
) -> gpd.GeoDataFrame:
    """Spatially join POIs to their nearest hexagon centroid."""
    if poi_gdf.empty:
        raise ValueError("poi_gdf is empty; cannot assign POIs to hexagons.")
    if hex_gdf.empty:
        raise ValueError("hex_gdf is empty; cannot assign POIs to hexagons.")

    if hex_id_col not in hex_gdf.columns:
        hex_gdf = hex_gdf.assign(**{hex_id_col: hex_gdf.index.astype(str)})

    poi_proj = poi_gdf.to_crs(projected_crs)
    hex_proj = hex_gdf[[hex_id_col, "geometry"]].to_crs(projected_crs)

    joined = gpd.sjoin_nearest(
        poi_proj,
        hex_proj,
        how="left",
        distance_col="nearest_dist",
    )
    joined = joined.rename(columns={f"{hex_id_col}_right": hex_id_col})
    joined = joined.drop(columns=["index_right"], errors="ignore")
    return joined.to_crs(poi_gdf.crs)


def _parse_coord_string(value) -> tuple[float, float]:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return (_to_float(value[0]), _to_float(value[1]))
    if not isinstance(value, str):
        return (np.nan, np.nan)

    try:
        parsed = ast.literal_eval(value)
    except Exception:
        return (np.nan, np.nan)

    if isinstance(parsed, (tuple, list)) and len(parsed) == 2:
        return (_to_float(parsed[0]), _to_float(parsed[1]))
    return (np.nan, np.nan)


def _is_valid_pair(value) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and all(isinstance(v, float) and not np.isnan(v) for v in value)
    )


def _to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _coerce_int(series: pd.Series) -> pd.Series:
    try:
        return series.astype(int)
    except ValueError:
        return series
