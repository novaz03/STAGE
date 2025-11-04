"""Aggregation helpers for spatial embeddings."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd

from shapely import wkb, wkt
from shapely.geometry import base as shapely_base

from .fill import fill_missing_vectors
from .parquet_utils import read_parquet_row_groups
from .data import assign_pois_to_hexagons

LOGGER = logging.getLogger(__name__)


def mean_vectors_by_group(
    df: pd.DataFrame,
    group_column: str,
    vector_column: str,
    count_column: str = "poi_count",
) -> pd.DataFrame:
    """
    Compute the mean vector for each group.

    Parameters
    ----------
    df : DataFrame
        Must contain ``group_column`` and ``vector_column`` where vectors are
        sequences convertible to numpy arrays.
    group_column : str
        Column name denoting the group ID (e.g., hexagon or block group).
    vector_column : str
        Column containing the vector representations (list/array-like).
    count_column : str
        Name of the column that stores the number of contributing items.

    Returns
    -------
    DataFrame with columns ``group_column``, ``vector_column`` and
    ``count_column`` where the vectors are averaged.
    """
    if group_column not in df.columns:
        raise ValueError(f"Missing group column '{group_column}'")
    if vector_column not in df.columns:
        raise ValueError(f"Missing vector column '{vector_column}'")

    valid = df.dropna(subset=[group_column, vector_column])
    if valid.empty:
        LOGGER.warning("No valid rows available for aggregation.")
        return pd.DataFrame(columns=[group_column, vector_column, count_column])

    def _mean_stack(values: Iterable) -> np.ndarray:
        vectors = [np.asarray(v, dtype=np.float32) for v in values]
        return np.stack(vectors).mean(axis=0)

    aggregated = valid.groupby(group_column)[vector_column].agg(_mean_stack).reset_index()
    counts = valid.groupby(group_column)[vector_column].size().rename(count_column).reset_index()
    return aggregated.merge(counts, on=group_column, how="left")


@dataclass
class AggregationConfig:
    """Configuration for aggregating POI embeddings to spatial cells."""

    poi_geometry_csv: Path
    poi_parquet: Path
    hex_parquet: Path
    output_path: Path
    poi_id_col: str = "PLACEKEY"
    hex_id_col: str = "hex_id"
    latent_column: str = "z_poi"
    geometry_col: str = "geometry"
    projected_crs: int = 5070
    count_column: str = "poi_count"


def aggregate_poi_latents_to_hex(config: AggregationConfig) -> gpd.GeoDataFrame:
    """
    Merge latent POI embeddings with geometry, assign to hexagons, and compute mean vectors.

    Returns the hexagon GeoDataFrame with the aggregated embedding column and writes the
    result to ``config.output_path``.
    """

    LOGGER.info("Loading POI geometry from %s", config.poi_geometry_csv)
    poi_df = pd.read_csv(config.poi_geometry_csv)
    if config.poi_id_col not in poi_df.columns:
        raise ValueError(f"{config.poi_id_col} column missing in POI geometry file.")
    if config.geometry_col in poi_df.columns:
        geom_series = poi_df[config.geometry_col]
        sample = geom_series.dropna().iloc[0] if not geom_series.dropna().empty else None
        if isinstance(sample, shapely_base.BaseGeometry):
            geometry = geom_series
        elif isinstance(sample, (bytes, bytearray)):
            geometry = gpd.GeoSeries.from_wkb(geom_series)
        elif isinstance(sample, str):
            try:
                geometry = geom_series.map(wkb.loads)
            except Exception:
                geometry = geom_series.map(wkt.loads)
        else:
            raise ValueError(f"Unsupported geometry type in {config.poi_geometry_csv}.")
        poi_gdf = gpd.GeoDataFrame(poi_df, geometry=geometry, crs="EPSG:4326")
    else:
        if not {"LONGITUDE", "LATITUDE"}.issubset(poi_df.columns):
            raise ValueError("POI geometry CSV must contain LONGITUDE and LATITUDE columns when geometry is absent.")
        poi_gdf = gpd.GeoDataFrame(
            poi_df,
            geometry=gpd.points_from_xy(poi_df["LONGITUDE"], poi_df["LATITUDE"]),
            crs="EPSG:4326",
        )

    LOGGER.info("Loading POI embeddings from %s", config.poi_parquet)
    poi_latent_df = read_parquet_row_groups(config.poi_parquet)
    if config.poi_id_col not in poi_latent_df.columns:
        raise ValueError(f"{config.poi_id_col} column missing in POI parquet.")
    if config.latent_column not in poi_latent_df.columns:
        raise ValueError(f"{config.latent_column} column missing in POI parquet.")

    merged = poi_gdf.merge(
        poi_latent_df[[config.poi_id_col, config.latent_column]],
        on=config.poi_id_col,
        how="inner",
    )
    LOGGER.info("Merged %d POIs with latent vectors", len(merged))

    hex_gdf = gpd.read_parquet(config.hex_parquet)
    if config.hex_id_col not in hex_gdf.columns:
        hex_gdf = hex_gdf.assign(**{config.hex_id_col: hex_gdf.index.astype(str)})
    if config.geometry_col not in hex_gdf.columns:
        raise ValueError(f"Hex tessellation must include a '{config.geometry_col}' column.")
    geom_hex = hex_gdf[config.geometry_col]
    sample_hex = geom_hex.dropna().iloc[0] if not geom_hex.dropna().empty else None
    if isinstance(sample_hex, shapely_base.BaseGeometry):
        geometry = geom_hex
    elif isinstance(sample_hex, (bytes, bytearray)):
        geometry = gpd.GeoSeries.from_wkb(geom_hex)
    elif isinstance(sample_hex, str):
        try:
            geometry = geom_hex.map(wkb.loads)
        except Exception:
            geometry = geom_hex.map(wkt.loads)
    else:
        raise ValueError("Unsupported geometry encoding in hex tessellation parquet.")
    hex_gdf = gpd.GeoDataFrame(hex_gdf, geometry=geometry, crs=poi_gdf.crs)

    joined = assign_pois_to_hexagons(
        merged, hex_gdf, poi_id_col=config.poi_id_col, hex_id_col=config.hex_id_col, projected_crs=config.projected_crs
    )
    if joined.empty:
        raise ValueError("No POIs were matched to hexagons; check the spatial inputs.")

    aggregated = mean_vectors_by_group(
        joined,
        group_column=config.hex_id_col,
        vector_column=config.latent_column,
        count_column=config.count_column,
    )

    hex_with_latents = hex_gdf.merge(aggregated, on=config.hex_id_col, how="left")
    if config.latent_column not in hex_with_latents.columns:
        hex_with_latents[config.latent_column] = None

    sample_latent = None
    for candidate in merged[config.latent_column]:
        if candidate is None:
            continue
        arr = np.asarray(candidate, dtype=np.float32)
        if arr.size:
            sample_latent = arr
            break
    if sample_latent is None:
        sample_latent = np.zeros(1, dtype=np.float32)

    hex_with_latents[config.latent_column] = fill_missing_vectors(
        hex_with_latents[config.latent_column],
        sample_latent.astype(np.float32),
    )

    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hex_with_latents.to_parquet(output_path, index=False)
    LOGGER.info("Wrote aggregated hex embeddings to %s", output_path)
    return hex_with_latents
