"""Data access helpers extracted from the exploratory notebooks.

The notebooks under ``src/evacmob/notebooks`` contain rich, but highly
stateful, data-access code.  This module refactors the reusable pieces into
pure functions that work with paths and buffers instead of relying on
notebook globals.  The helpers below mirror the behaviour of the functions
used in the ``cbgses`` and ``Report_results`` notebooks while keeping the
API light-weight and testable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import wkt
from shapely.geometry import base as shapely_base

LOGGER = logging.getLogger(__name__)


def load_parquet(path: str | Path) -> pd.DataFrame:
    """Read a parquet file into a :class:`pandas.DataFrame`."""
    return pd.read_parquet(path)


def save_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    """Persist a dataframe to parquet, creating parent directories if missing."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path


def _coerce_geometry(
    series: pd.Series,
) -> gpd.GeoSeries:
    """Best-effort conversion of string/bytes geometry columns to Shapely objects."""
    sample = series.dropna().iloc[0] if not series.dropna().empty else None
    if isinstance(sample, shapely_base.BaseGeometry):
        return gpd.GeoSeries(series)
    if isinstance(sample, (bytes, bytearray)):
        return gpd.GeoSeries.from_wkb(series)
    if isinstance(sample, str):
        return gpd.GeoSeries(series.map(wkt.loads))
    return gpd.GeoSeries(series)


def load_hexagon_data(
    path: str | Path,
    geometry_col: str = "geometry",
    crs: str | int | None = "EPSG:4326",
    id_col: str = "hex_id",
    drop_columns: Sequence[str] = ("hexid",),
) -> gpd.GeoDataFrame:
    """Load the SES hexagon parquet used throughout the notebooks."""
    LOGGER.debug("Loading hexagon data from %s", path)
    gdf = gpd.read_parquet(path)

    # Reset the geometry if it is stored as WKB/WKT.
    if not isinstance(gdf[geometry_col].iloc[0], shapely_base.BaseGeometry):
        gdf[geometry_col] = _coerce_geometry(gdf[geometry_col])
    gdf = gpd.GeoDataFrame(gdf, geometry=geometry_col, crs=crs)

    for col in drop_columns:
        if col in gdf.columns:
            gdf = gdf.drop(columns=col)

    if id_col not in gdf.columns:
        gdf[id_col] = gdf.index.astype(str)

    return gdf.reset_index(drop=True)


def load_poi_data(
    path: str | Path,
    geometry_col: str = "geometry",
    wkt_fallback: bool = True,
    lat_col_candidates: Iterable[str] = ("lat", "latitude", "LATITUDE"),
    lon_col_candidates: Iterable[str] = ("lon", "longitude", "LONGITUDE"),
    crs: str | int | None = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """Load a POI CSV exported in the notebooks and coerce to GeoDataFrame."""
    LOGGER.debug("Loading POI data from %s", path)
    df = pd.read_csv(path)

    if geometry_col in df.columns:
        geom_series = df[geometry_col]
        if not isinstance(geom_series.dropna().iloc[0], shapely_base.BaseGeometry):
            geom_series = _coerce_geometry(geom_series)
    elif wkt_fallback:
        lat_col = _resolve_column(df, lat_col_candidates)
        lon_col = _resolve_column(df, lon_col_candidates)
        geom_series = gpd.points_from_xy(df[lon_col], df[lat_col], crs=crs)
    else:
        raise ValueError(
            "Could not determine geometry column; specify geometry_col or disable wkt_fallback."
        )

    gdf = gpd.GeoDataFrame(df, geometry=geom_series, crs=crs)
    return gdf


def parse_vector_column(series: pd.Series) -> np.ndarray:
    """Convert a column of notebook-style vectors into a 2D numpy array."""

    def _parse_single(value: object) -> np.ndarray:
        if isinstance(value, (list, tuple, np.ndarray)):
            return np.asarray(value, dtype=np.float32)
        if isinstance(value, str):
            cleaned = value.strip("[]")
            # The notebook stored arrays either space or comma separated.
            sep = "," if "," in cleaned else " "
            tokens = [t for t in cleaned.split(sep) if t.strip()]
            return np.asarray([float(t) for t in tokens], dtype=np.float32)
        raise TypeError(f"Unsupported vector type: {type(value)!r}")

    vectors = [_parse_single(v) for v in series]
    return np.vstack(vectors)


def assign_pois_to_hexagons(
    poi_gdf: gpd.GeoDataFrame,
    hex_gdf: gpd.GeoDataFrame,
    poi_id_col: str = "poi_id",
    hex_id_col: str = "hex_id",
    projected_crs: str | int = 5070,
) -> gpd.GeoDataFrame:
    """Spatially join POIs to their nearest hexagon."""
    if poi_gdf.empty or hex_gdf.empty:
        raise ValueError("Both poi_gdf and hex_gdf must be non-empty GeoDataFrames.")

    poi_proj = poi_gdf.to_crs(projected_crs)
    hex_proj = hex_gdf[[hex_id_col, "geometry"]].to_crs(projected_crs)

    joined = gpd.sjoin_nearest(
        poi_proj,
        hex_proj,
        how="left",
        distance_col="nearest_dist",
    )

    joined = joined.rename(columns={hex_id_col + "_right": hex_id_col})
    joined = joined.drop(columns=["index_right"], errors="ignore")

    LOGGER.debug(
        "Matched %d / %d POIs to hexagons",
        joined[hex_id_col].notna().sum(),
        len(joined),
    )

    # Restore original CRS for downstream operations.
    joined = joined.to_crs(poi_gdf.crs)
    return joined


def _resolve_column(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    """Resolve a column name from a list of case-insensitive candidates."""
    cols = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in cols:
            return cols[name.lower()]
    raise KeyError(f"None of {list(candidates)} found in columns: {list(df.columns)}")
