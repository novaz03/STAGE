"""Preprocessing helpers ported from the exploratory notebooks."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import geopandas as gpd
from pandas.api.types import is_numeric_dtype
from shapely.geometry import Point, LineString
from pyproj import Geod


LOGGER = logging.getLogger(__name__)
_GEOD = Geod(ellps="WGS84")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Scale numeric columns to zero mean / unit variance."""
    numeric_cols = [c for c in df.columns if is_numeric_dtype(df[c])]
    if not numeric_cols:
        LOGGER.debug("No numeric columns found for normalization.")
        return df
    result = df.copy()
    for col in numeric_cols:
        series = result[col]
        mean = series.mean()
        std = series.std(ddof=0)
        if std == 0 or np.isnan(std):
            continue
        result[col] = (series - mean) / std
    return result


def make_points_gdf(
    df: pd.DataFrame,
    start_lon_col: str = "tripStartLongitude",
    start_lat_col: str = "tripStartLatitude",
    end_lon_col: str = "tripEndLongitude",
    end_lat_col: str = "tripEndLatitude",
    start_time_col: str = "tripStartDate",
    end_time_col: str = "tripEndDate",
    crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """Construct a GeoDataFrame of trip start points and keep end points as columns."""
    df = df.copy()
    for col in (start_time_col, end_time_col):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    def _maybe_point(lon, lat):
        return Point(lon, lat) if pd.notnull(lon) and pd.notnull(lat) else None

    df["start_geom"] = [
        _maybe_point(lon, lat) for lon, lat in zip(df[start_lon_col], df[start_lat_col])
    ]
    df["end_geom"] = [_maybe_point(lon, lat) for lon, lat in zip(df[end_lon_col], df[end_lat_col])]

    gdf = gpd.GeoDataFrame(df, geometry="start_geom", crs=crs)
    gdf = gdf.dropna(subset=["start_geom", "end_geom", start_time_col, end_time_col])
    return gdf


def _geo_dist_m(p1: Point, p2: Point) -> float:
    """Great-circle distance between two points in metres."""
    _, _, dist_m = _GEOD.inv(p1.x, p1.y, p2.x, p2.y)
    return dist_m


@dataclass
class TripSegments:
    """Outputs from :func:`stitch_trips_to_lines_with_gaps`."""

    segments: gpd.GeoDataFrame
    links: gpd.GeoDataFrame


def stitch_trips_to_lines_with_gaps(
    gdf_points: gpd.GeoDataFrame,
    person_col: str = "participantId",
    join_dist_m: float = 100.0,
    join_gap_max: pd.Timedelta = pd.Timedelta("4H"),
    hard_break_gap: pd.Timedelta = pd.Timedelta("8H"),
) -> TripSegments:
    """Recreate the trip stitching logic from ``DRIVES_traj.ipynb``."""
    cols_needed = [
        person_col,
        "tripStartDate",
        "tripEndDate",
        "start_geom",
        "end_geom",
    ]
    missing = [c for c in cols_needed if c not in gdf_points.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    seg_rows = []
    link_rows = []

    for pid, df_person in gdf_points.groupby(person_col):
        dfp = df_person.sort_values("tripStartDate").reset_index(drop=True)

        seg_id = 0
        seg_points: list[Point] = []
        seg_trips: list[int] = []
        seg_start_time = None
        seg_end_time = None
        last_end_point = None
        last_end_time = None
        seg_gaps: list[float] = []
        link_idx = 0

        def flush_segment(tag: str) -> None:
            nonlocal seg_id, seg_points, seg_trips, seg_start_time, seg_end_time
            nonlocal seg_gaps, link_idx

            if len(seg_points) >= 2:
                link_count = len(seg_gaps)
                if link_count:
                    max_gap_h = max(seg_gaps) / 3600.0
                    mean_gap_h = (sum(seg_gaps) / link_count) / 3600.0
                    total_gap_h = sum(seg_gaps) / 3600.0
                else:
                    max_gap_h = mean_gap_h = total_gap_h = 0.0

                seg_rows.append(
                    {
                        person_col: pid,
                        "seg_id": seg_id,
                        "num_trips": len(seg_trips),
                        "link_count": link_count,
                        "start_time": seg_start_time,
                        "end_time": seg_end_time,
                        "max_gap_h": max_gap_h,
                        "mean_gap_h": mean_gap_h,
                        "total_gap_h": total_gap_h,
                        "segment_type": tag,
                        "geometry": LineString(seg_points),
                    }
                )

            seg_id += 1
            seg_points = []
            seg_trips = []
            seg_gaps = []
            link_idx = 0

        for idx, row in dfp.iterrows():
            start_pt: Point = row["start_geom"]
            end_pt: Point = row["end_geom"]
            start_time = row["tripStartDate"]
            end_time = row["tripEndDate"]

            if last_end_point is None:
                seg_points = [start_pt, end_pt]
                seg_trips = [idx]
                seg_start_time = start_time
                seg_end_time = end_time
                last_end_point = end_pt
                last_end_time = end_time
                continue

            gap = start_time - last_end_time
            dist = _geo_dist_m(last_end_point, start_pt)

            if pd.notnull(gap) and gap <= join_gap_max and dist <= join_dist_m:
                seg_points.extend([start_pt, end_pt])
                seg_trips.append(idx)
                seg_end_time = end_time
                gap_seconds = gap.total_seconds()
                seg_gaps.append(gap_seconds)
                link_rows.append(
                    {
                        person_col: pid,
                        "seg_id": seg_id,
                        "link_idx": link_idx,
                        "stop_time": last_end_time,
                        "next_start_time": start_time,
                        "gap_s": gap_seconds,
                        "gap_h": gap_seconds / 3600.0,
                        "geometry": LineString([last_end_point, start_pt]),
                    }
                )
                link_idx += 1
            else:
                tag = "hard_break" if gap > hard_break_gap else "gap_exceeded"
                flush_segment(tag)
                seg_points = [start_pt, end_pt]
                seg_trips = [idx]
                seg_start_time = start_time
                seg_end_time = end_time
                seg_gaps = []

            last_end_point = end_pt
            last_end_time = end_time

        flush_segment("final")

    gdf_segments = gpd.GeoDataFrame(seg_rows, geometry="geometry", crs=gdf_points.crs)
    gdf_links = gpd.GeoDataFrame(link_rows, geometry="geometry", crs=gdf_points.crs)
    return TripSegments(segments=gdf_segments, links=gdf_links)


def _resolve_col(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    """Case-insensitive column resolution used throughout the notebooks."""
    cols = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in cols:
            return cols[name.lower()]
    raise KeyError(f"None of {list(candidates)} exist in {list(df.columns)}")


def _haversine_m(lat1, lon1, lat2, lon2):
    """Vectorised haversine used in ``nearest_pois_for_links``."""
    R = 6_371_000.0
    lat1 = np.deg2rad(lat1)
    lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2)
    lon2 = np.deg2rad(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return R * c


def nearest_pois_for_links(
    gdf_links: gpd.GeoDataFrame,
    poi_df: pd.DataFrame,
    k: int = 10,
    n_links: int | None = None,
    person_col: str = "participantId",
) -> pd.DataFrame:
    """Mirror ``nearest_pois_for_links`` from the DRIVES notebook."""
    n_links = n_links or len(gdf_links)

    lat_col = _resolve_col(poi_df, ["LATITUDE", "latitude", "lat"])
    lon_col = _resolve_col(poi_df, ["LONGITUDE", "longitude", "lon", "lng"])
    name_col = _resolve_col(poi_df, ["LOCATION_NAME", "location_name", "name"])
    top_col = _resolve_col(poi_df, ["TOP_CATEGORY", "top_category"])
    sub_col = _resolve_col(poi_df, ["SUB_CATEGORY", "sub_category"])
    key_col = _resolve_col(poi_df, ["PLACEKEY", "placekey"])

    pois = poi_df.dropna(subset=[lat_col, lon_col]).copy()
    poi_lat = pois[lat_col].to_numpy(dtype=float)
    poi_lon = pois[lon_col].to_numpy(dtype=float)

    out_rows = []
    links = gdf_links.iloc[:n_links].reset_index(drop=True)

    for i, row in links.iterrows():
        geom = row.geometry
        if not isinstance(geom, LineString) or geom.is_empty:
            continue

        start_lon, start_lat = geom.coords[0]
        end_lon, end_lat = geom.coords[-1]

        for endpoint, (qlat, qlon) in (
            ("start", (start_lat, start_lon)),
            ("end", (end_lat, end_lon)),
        ):
            dists = _haversine_m(qlat, qlon, poi_lat, poi_lon)
            eff = min(k, dists.size)
            idx_k = np.argpartition(dists, eff - 1)[:eff]
            idx_sorted = idx_k[np.argsort(dists[idx_k])]

            for rank, j in enumerate(idx_sorted, start=1):
                poi_row = pois.iloc[j]
                out_rows.append(
                    {
                        person_col: row.get(person_col),
                        "seg_id": row.get("seg_id"),
                        "link_idx": row.get("link_idx", i),
                        "endpoint": endpoint,
                        "rank": rank,
                        "poi_placekey": poi_row[key_col],
                        "poi_name": poi_row[name_col],
                        "top_category": poi_row[top_col],
                        "sub_category": poi_row[sub_col],
                        "poi_lat": float(poi_row[lat_col]),
                        "poi_lon": float(poi_row[lon_col]),
                        "distance_m": float(dists[j]),
                        "q_lat": float(qlat),
                        "q_lon": float(qlon),
                    }
                )

    result = pd.DataFrame(out_rows).sort_values(
        [person_col, "seg_id", "link_idx", "endpoint", "rank"], kind="stable"
    )
    return result.reset_index(drop=True)


def iterative_impute(
    df: pd.DataFrame,
    exempt_columns: Iterable[str] | None = None,
    random_state: int | None = 0,
) -> pd.DataFrame:
    """Apply :class:`sklearn.impute.IterativeImputer` mirroring the notebook code."""
    try:
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "scikit-learn is required for iterative_impute; install 'scikit-learn'."
        ) from exc

    exempt = list(exempt_columns or [])
    to_impute = df.columns.difference(exempt)

    imp = IterativeImputer(random_state=random_state)
    imputed_vals = imp.fit_transform(df[to_impute])

    df_imputed = pd.DataFrame(imputed_vals, columns=to_impute, index=df.index)
    result = pd.concat([df[exempt], df_imputed], axis=1)[df.columns]
    return result
