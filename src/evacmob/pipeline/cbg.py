"""Utilities for building census block group (CBG) socio-economic surfaces."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
import geopandas as gpd
import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass
class CBGLoadConfig:
    """Configuration to load attribute and geometry tables for CBG SES."""

    attributes_path: Path = Path("bg_fl_2022.xlsx")
    geometry_path: Path = Path("fl_bg.geojson")
    geometry_crs: str = "EPSG:4326"
    merge_key: str = "GEOID"


def load_cbgses(config: CBGLoadConfig) -> gpd.GeoDataFrame:
    """
    Load the SES attribute table and geometry and produce a merged GeoDataFrame.
    """
    LOGGER.info("Loading CBG attributes from %s", config.attributes_path)
    attrs = pd.read_excel(config.attributes_path)

    LOGGER.info("Loading CBG geometries from %s", config.geometry_path)
    geom = gpd.read_file(config.geometry_path)
    if config.geometry_crs:
        geom = geom.to_crs(config.geometry_crs)

    attrs[config.merge_key] = attrs[config.merge_key].astype(str)
    geom[config.merge_key] = geom[config.merge_key].astype(str)

    LOGGER.info("Merging attribute and geometry tables")
    merged = geom.merge(attrs, on=config.merge_key, how="left")
    LOGGER.info("Merged CBG GeoDataFrame with %s rows", len(merged))
    return merged
