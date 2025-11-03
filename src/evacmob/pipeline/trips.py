"""Trip-log preprocessing helpers for the evacmob pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import geopandas as gpd

from .. import preprocess as preprocess_mod


def load_trip_logs(path: str | Path, fmt: str = "auto") -> pd.DataFrame:
    """Load raw trip logs from CSV/Parquet."""
    path = Path(path)
    if fmt == "auto":
        fmt = path.suffix.lower().lstrip(".") or "csv"
    if fmt in {"csv", "tsv"}:
        sep = "," if fmt == "csv" else "\t"
        return pd.read_csv(path, sep=sep)
    if fmt in {"parquet", "pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported trip log format '{fmt}'.")


def preprocess_trip_logs(
    trip_df: pd.DataFrame,
    person_col: str,
    join_dist_m: float,
    join_gap_hours: float,
    hard_break_hours: float,
) -> preprocess_mod.TripSegments:
    """Convert raw trip logs into stitched trajectory segments and links."""
    gdf_points = preprocess_mod.make_points_gdf(trip_df)
    segments = preprocess_mod.stitch_trips_to_lines_with_gaps(
        gdf_points,
        person_col=person_col,
        join_dist_m=join_dist_m,
        join_gap_max=pd.Timedelta(hours=join_gap_hours),
        hard_break_gap=pd.Timedelta(hours=hard_break_hours),
    )
    return segments


def build_trajectory_features_from_segments(
    segments: gpd.GeoDataFrame,
    person_col: str,
    id_col: str = "traj_id",
) -> pd.DataFrame:
    """Create a numeric feature table from stitched segments."""
    if segments.empty:
        return pd.DataFrame(columns=[id_col])
    numeric_cols = segments.select_dtypes(include=["number"]).columns.tolist()
    feature_df = segments.drop(columns=["geometry"], errors="ignore").copy()
    feature_df[id_col] = (
        feature_df[person_col].astype(str) + "_" + feature_df["seg_id"].astype(str)
    )
    cols = [id_col] + [c for c in numeric_cols if c != "seg_id"]
    feature_df = feature_df[cols].fillna(0.0)
    return feature_df
