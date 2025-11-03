"""Visualization helpers (folium, matplotlib)."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


def copy_static_report(src_html: str | Path, dest: str | Path) -> Path:
    src = Path(src_html)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(src.read_bytes())
    return dest


def plot_hexgrid_with_inset(
    hex_gdf: gpd.GeoDataFrame,
    ses_gdf: gpd.GeoDataFrame,
    states_shapefile: str | Path,
    inset_bounds: Tuple[float, float, float, float],
    out_path: str | Path,
    figsize: Tuple[int, int] = (20, 20),
) -> Path:
    """Recreate the hex-grid plot (with inset) from ``cbgses.ipynb``."""
    states = gpd.read_file(states_shapefile)
    fig, ax = plt.subplots(figsize=figsize)

    hex_gdf.boundary.plot(ax=ax, linewidth=0.05, edgecolor="gray", alpha=0.5)
    states[states.STUSPS == "FL"].boundary.plot(ax=ax, linewidth=1, color="black")
    ses_gdf.plot(ax=ax, cmap="tab20", alpha=0.9)

    xmin, ymin, xmax, ymax = inset_bounds
    axins = inset_axes(ax, width="50%", height="50%", loc="lower left", borderpad=1)
    axins.set_xlim(xmin, xmax)
    axins.set_ylim(ymin, ymax)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="red", lw=1)

    hex_gdf.boundary.plot(ax=axins, linewidth=0.1, edgecolor="black", alpha=0.9)
    states[states.STUSPS == "FL"].boundary.plot(ax=axins, linewidth=1, color="black")
    ses_gdf.plot(ax=axins, cmap="tab20", alpha=0.9)

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path
