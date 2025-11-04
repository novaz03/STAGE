"""Geometry-related helpers for spatial preprocessing."""

from __future__ import annotations

import logging

import geopandas as gpd
from math import sqrt, cos, sin, pi
from typing import Tuple

from shapely.geometry import MultiPoint, Polygon

LOGGER = logging.getLogger(__name__)


def build_concave_hull(points: gpd.GeoDataFrame, ratio: float = 0.05) -> Polygon:
    """
    Compute a concave hull around the provided point GeoDataFrame.

    Parameters
    ----------
    points:
        GeoDataFrame containing point geometries. CRS is assumed to be EPSG:4326.
    ratio:
        Controls the concavity of the hull. Mirrors the notebook default (0.05).
    """
    if points.empty:
        raise ValueError("Point GeoDataFrame is empty; cannot build concave hull.")

    multipoint = MultiPoint(points.geometry.tolist())
    hull_series = gpd.GeoSeries([multipoint], crs=points.crs)
    hull = hull_series.concave_hull(ratio=ratio, allow_holes=False).iloc[0]
    LOGGER.info("Computed concave hull with ratio %s", ratio)
    return hull


def load_hexagon_grid_placeholder(*_, **__) -> gpd.GeoDataFrame:
    """
    Placeholder for future hexagon grid loading logic.

    The hex grid integration from the notebook has not yet been modularised; this
    function raises ``NotImplementedError`` to make the pending work explicit.
    """
    raise NotImplementedError(
        "Hexagon grid loading is not yet implemented in the modular pipeline."
    )


def _pointy_hex_vertices(cx: float, cy: float, r: float) -> list[Tuple[float, float]]:
    """Return 6 vertices for a pointy-topped hexagon centered at (cx, cy)."""
    angles = [pi / 2, 5 * pi / 6, 7 * pi / 6, 3 * pi / 2, 11 * pi / 6, pi / 6]
    return [(cx + r * cos(a), cy + r * sin(a)) for a in angles]


def generate_hex_grid_over_polygon(
    polygon: Polygon,
    hex_radius_m: float,
    crs_source: str = "EPSG:4326",
    crs_projected: str = "EPSG:5070",
    filter_mode: str = "intersects",
) -> gpd.GeoDataFrame:
    """
    Generate a non-overlapping pointy-topped hexagon grid covering a polygon.

    - Hexagons are generated in a projected CRS for metric accuracy (default EPSG:5070).
    - Returned GeoDataFrame is in the source CRS (default EPSG:4326).
    - filter_mode: "intersects" keeps any hex intersecting the polygon; "centroid"
      keeps hexes whose centroid lies within the polygon.
    """
    if hex_radius_m <= 0:
        raise ValueError("hex_radius_m must be positive")

    # Prepare polygon in the projected CRS
    hull_gdf = gpd.GeoDataFrame({"geometry": [polygon]}, crs=crs_source)
    hull_proj = hull_gdf.to_crs(crs_projected)
    poly_proj = hull_proj.geometry.iloc(0)
    try:
        poly_proj = hull_proj.geometry.iloc[0]
    except Exception:
        poly_proj = hull_proj.geometry.values[0]

    minx, miny, maxx, maxy = poly_proj.bounds

    # Pointy-topped hex grid spacing
    r = float(hex_radius_m)
    dx = sqrt(3.0) * r  # horizontal distance between centers
    dy = 1.5 * r  # vertical distance between centers

    hex_polys = []
    row = 0
    y = miny - r
    while y <= maxy + r:
        # offset every other row by half the horizontal spacing
        x_offset = 0.5 * dx if (row % 2) else 0.0
        x = (minx - dx) + x_offset
        while x <= (maxx + dx):
            vertices = _pointy_hex_vertices(x, y, r)
            hex_poly = Polygon(vertices)
            if filter_mode == "centroid":
                if poly_proj.contains(hex_poly.centroid):
                    hex_polys.append(hex_poly)
            else:
                if poly_proj.intersects(hex_poly):
                    hex_polys.append(hex_poly)
            x += dx
        row += 1
        y += dy

    hex_gdf_proj = gpd.GeoDataFrame({"geometry": hex_polys}, crs=crs_projected)
    hex_gdf = hex_gdf_proj.to_crs(crs_source)
    hex_gdf = hex_gdf.reset_index(drop=True)
    hex_gdf["hex_id"] = hex_gdf.index.astype(str)
    hex_gdf["centroid"] = hex_gdf.centroid
    hex_gdf["centroid_x"] = hex_gdf["centroid"].x
    hex_gdf["centroid_y"] = hex_gdf["centroid"].y
    return hex_gdf


def check_hex_non_overlap(
    hex_gdf: gpd.GeoDataFrame,
    crs_projected: str = "EPSG:5070",
    tolerance_area: float = 1e-6,
) -> tuple[bool, int]:
    """
    Validate that generated hexagons do not overlap (beyond a tiny tolerance).

    Returns (is_valid, overlap_pairs) where overlap_pairs is the count of pairs
    whose intersection area exceeds the tolerance.
    """
    if hex_gdf.empty:
        return True, 0
    # Project for stable area computations
    proj = hex_gdf.to_crs(crs_projected)
    sindex = proj.sindex
    overlaps = 0
    geoms = proj.geometry.values
    for i, gi in enumerate(geoms):
        for j in sindex.intersection(gi.bounds):
            if j <= i:
                continue
            inter = gi.intersection(geoms[j])
            if inter.is_empty:
                continue
            if inter.area > tolerance_area:
                overlaps += 1
    return overlaps == 0, overlaps
