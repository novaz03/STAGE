"""Utilities to generate filtered POI CSV and hex tessellation parquet."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, Tuple

import geopandas as gpd
import pandas as pd

from .geometry import build_concave_hull, generate_hex_grid_over_polygon

LOGGER = logging.getLogger(__name__)


@dataclass
class FilteredPOIConfig:
    """Configuration for producing the study-area POI CSV."""

    input_csv: Path
    output_csv: Path = Path("Hex_bound_POI.csv")
    longitude_column: str = "LONGITUDE"
    latitude_column: str = "LATITUDE"
    geometry_column: str | None = None
    text_columns: Sequence[str] = field(
        default_factory=lambda: ("TOP_CATEGORY", "SUB_CATEGORY", "LOCATION_NAME")
    )
    columns_to_keep: Sequence[str] = field(
        default_factory=lambda: (
            "PLACEKEY",
            "LONGITUDE",
            "LATITUDE",
            "concatenated",
            "REGION",
            "LOCATION_NAME",
        )
    )
    bounding_box: Tuple[float, float, float, float] = (-88.57, -79.95, 24.45, 32.35)


def generate_hex_bound_poi(config: FilteredPOIConfig) -> gpd.GeoDataFrame:
    """Filter the raw POI export to the study bounding box and persist to CSV."""

    LOGGER.info("Loading raw POIs from %s", config.input_csv)
    df = pd.read_csv(config.input_csv)

    if config.geometry_column and config.geometry_column in df.columns:
        geometries = gpd.GeoSeries.from_wkt(df[config.geometry_column].dropna())
        df = df.loc[geometries.index].reset_index(drop=True)
        poi_gdf = gpd.GeoDataFrame(
            df,
            geometry=geometries.reset_index(drop=True),
            crs="EPSG:4326",
        )
    else:
        for column in config.text_columns:
            if column in df.columns:
                df[column] = df[column].fillna("<null_val>")
        if "concatenated" not in df.columns:
            missing = [c for c in config.text_columns if c not in df.columns]
            if missing:
                raise ValueError(
                    "Unable to compute 'concatenated' column; missing: " + ", ".join(missing)
                )
            df["concatenated"] = (
                df[config.text_columns[0]].astype(str)
                + "[sep]"
                + df[config.text_columns[1]].astype(str)
                + "[sep]"
                + df[config.text_columns[2]].astype(str)
            )
        min_lon, max_lon, min_lat, max_lat = config.bounding_box
        mask = (
            (df[config.longitude_column] >= min_lon)
            & (df[config.longitude_column] <= max_lon)
            & (df[config.latitude_column] >= min_lat)
            & (df[config.latitude_column] <= max_lat)
        )
        filtered = df.loc[mask, config.columns_to_keep].copy()
        poi_gdf = gpd.GeoDataFrame(
            filtered,
            geometry=gpd.points_from_xy(
                filtered[config.longitude_column], filtered[config.latitude_column]
            ),
            crs="EPSG:4326",
        )

    poi_gdf.to_csv(config.output_csv, index=False)
    LOGGER.info("Wrote %s filtered POIs to %s", len(poi_gdf), config.output_csv)
    return poi_gdf


@dataclass
class HexTessellationConfig:
    """Configuration for generating the hex tessellation parquet."""

    poi_csv: Path = Path("Hex_bound_POI.csv")
    output_parquet: Path = Path("Hex_tesse_raw.parquet")
    longitude_column: str = "LONGITUDE"
    latitude_column: str = "LATITUDE"
    concave_ratio: float = 0.05
    hex_radius_m: float = 8000.0
    filter_mode: str = "intersects"


def generate_hex_tessellation(config: HexTessellationConfig) -> gpd.GeoDataFrame:
    """
    Build a concave hull from the filtered POIs and tessellate it with hexagons.
    """

    LOGGER.info("Loading filtered POIs from %s", config.poi_csv)
    df = pd.read_csv(config.poi_csv)
    if df.empty:
        raise ValueError("Filtered POI CSV is empty; cannot build tessellation.")

    poi_gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[config.longitude_column], df[config.latitude_column]),
        crs="EPSG:4326",
    )
    hull = build_concave_hull(poi_gdf[["geometry"]], ratio=config.concave_ratio)
    hex_gdf = generate_hex_grid_over_polygon(
        hull,
        hex_radius_m=config.hex_radius_m,
        filter_mode=config.filter_mode,
    )
    hex_gdf.to_parquet(config.output_parquet, index=False)
    LOGGER.info("Wrote %s hexes to %s", len(hex_gdf), config.output_parquet)
    return hex_gdf
