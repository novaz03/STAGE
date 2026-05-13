"""Clustering helpers for labeled simulation subsets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

from .simulation_results import SimulationResultsView, evaluate_simulation_results

DEFAULT_LABEL_COLUMN = "reference_lab"
DEFAULT_ID_COLUMN = "traj_id"
DEFAULT_EXCLUDE_COLUMNS = {
    "person_id",
    DEFAULT_ID_COLUMN,
    "cats",
    "traj_cluster",
    "traj_cluster_old",
    "cluster_id_bestk",
    "knn_k3_label_raw",
    "cluster_labelled_type",
    DEFAULT_LABEL_COLUMN,
}


@dataclass(frozen=True)
class ClusteringSearchResult:
    """Best clustering run found during a hyperparameter search."""

    view: SimulationResultsView
    k: int
    seed: int
    score_name: str
    score_value: float
    feature_columns: list[str]
    leaderboard: pd.DataFrame


@dataclass(frozen=True)
class FrechetDbscanResult:
    """Best Frechet+DBSCAN baseline found during percentile search."""

    view: SimulationResultsView
    eps_percentile: int
    eps: float
    min_samples: int
    score_name: str
    score_value: float
    ari: float
    nmi: float
    mapped_accuracy_all: float
    mapped_accuracy_non_noise: float
    n_clusters: int
    noise_fraction: float
    leaderboard: pd.DataFrame


def prepare_trajectory_feature_table(
    trajectories: pd.DataFrame,
    *,
    label_col: str = DEFAULT_LABEL_COLUMN,
    id_col: str = DEFAULT_ID_COLUMN,
    exclude_columns: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    """Prepare a numeric feature matrix from a trajectory-level CSV."""
    if label_col not in trajectories.columns:
        raise KeyError(f"'{label_col}' not found in trajectories.")
    if id_col not in trajectories.columns:
        raise KeyError(f"'{id_col}' not found in trajectories.")

    excluded = set(DEFAULT_EXCLUDE_COLUMNS)
    if exclude_columns:
        excluded.update(exclude_columns)

    df = trajectories.copy()
    for column in df.columns:
        if pd.api.types.is_bool_dtype(df[column]):
            df[column] = df[column].astype(int)

    candidate_columns = [col for col in df.columns if col not in excluded]
    numeric = df[candidate_columns].apply(pd.to_numeric, errors="coerce")
    usable = [col for col in numeric.columns if numeric[col].notna().any()]
    if not usable:
        raise ValueError("No usable numeric feature columns were found for clustering.")

    features = numeric[usable].copy()
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median(numeric_only=True)).fillna(0.0)

    X = StandardScaler().fit_transform(features.to_numpy(dtype=float))
    y_true = df[label_col].to_numpy(dtype=object)
    traj_ids = df[id_col].astype(str).to_numpy(dtype=object)
    return features, X, y_true, list(features.columns), traj_ids


def _trajectory_point_stats(group: pd.DataFrame) -> pd.Series:
    ordered = group.sort_values("pt_idx")
    lat = ordered["latitude"].to_numpy(dtype=float)
    lon = ordered["longitude"].to_numpy(dtype=float)

    lat_diff = np.diff(lat)
    lon_diff = np.diff(lon)
    step_deg = np.sqrt(lat_diff**2 + lon_diff**2)

    return pd.Series(
        {
            "n_points": len(ordered),
            "lat_mean": float(np.mean(lat)),
            "lat_std": float(np.std(lat)),
            "lat_min": float(np.min(lat)),
            "lat_max": float(np.max(lat)),
            "lon_mean": float(np.mean(lon)),
            "lon_std": float(np.std(lon)),
            "lon_min": float(np.min(lon)),
            "lon_max": float(np.max(lon)),
            "start_lat": float(lat[0]),
            "start_lon": float(lon[0]),
            "end_lat": float(lat[-1]),
            "end_lon": float(lon[-1]),
            "net_disp_deg": float(np.sqrt((lat[-1] - lat[0]) ** 2 + (lon[-1] - lon[0]) ** 2)),
            "total_step_deg": float(np.sum(step_deg)) if step_deg.size else 0.0,
            "max_step_deg": float(np.max(step_deg)) if step_deg.size else 0.0,
            "lat_span": float(np.max(lat) - np.min(lat)),
            "lon_span": float(np.max(lon) - np.min(lon)),
        }
    )


def prepare_point_feature_table(
    points: pd.DataFrame,
    *,
    label_col: str = DEFAULT_LABEL_COLUMN,
    id_col: str = DEFAULT_ID_COLUMN,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Aggregate point-level parquet rows into one feature row per trajectory."""
    required = {id_col, "pt_idx", "latitude", "longitude", label_col}
    missing = sorted(required.difference(points.columns))
    if missing:
        raise KeyError(f"Missing required point columns: {missing}")

    ordered = points.sort_values([id_col, "pt_idx"]).copy()
    labels = ordered.groupby(id_col, sort=False)[label_col].first()
    features = ordered.groupby(id_col, sort=False).apply(_trajectory_point_stats).reset_index()
    merged = features.merge(labels.rename(label_col), on=id_col, how="left", validate="one_to_one")

    feature_columns = [col for col in merged.columns if col not in {id_col, label_col}]
    X = StandardScaler().fit_transform(merged[feature_columns].to_numpy(dtype=float))
    y_true = merged[label_col].to_numpy(dtype=object)
    traj_ids = merged[id_col].astype(str).to_numpy(dtype=object)
    return merged, X, y_true, feature_columns, traj_ids


def load_clustering_input(
    path: str | Path,
    *,
    label_col: str = DEFAULT_LABEL_COLUMN,
    id_col: str = DEFAULT_ID_COLUMN,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Load either trajectory CSV or point parquet into a clustering-ready matrix."""
    input_path = Path(path)
    suffix = input_path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(input_path)
        features, X, y_true, feature_columns, traj_ids = prepare_trajectory_feature_table(
            df,
            label_col=label_col,
            id_col=id_col,
        )
        return df.assign(**{col: features[col] for col in feature_columns}), X, y_true, feature_columns, traj_ids

    if suffix in {".parquet", ".pq"}:
        gdf = gpd.read_parquet(input_path)
        return prepare_point_feature_table(gdf, label_col=label_col, id_col=id_col)

    raise ValueError(f"Unsupported input type '{suffix}'. Expected .csv or .parquet.")


def search_best_kmeans(
    X: np.ndarray,
    y_true: Sequence[object],
    *,
    k_values: Sequence[int],
    seeds: Sequence[int],
    score_name: str = "accuracy",
) -> ClusteringSearchResult:
    """Run repeated KMeans and return the best aligned clustering result."""
    if score_name not in {"accuracy", "balanced_accuracy"}:
        raise ValueError("score_name must be 'accuracy' or 'balanced_accuracy'.")

    leaderboard_rows: list[dict[str, float | int]] = []
    best: ClusteringSearchResult | None = None

    X_arr = np.asarray(X, dtype=float)
    y_true_arr = np.asarray(y_true, dtype=object)

    for k in k_values:
        for seed in seeds:
            labels = KMeans(n_clusters=k, random_state=seed, n_init=20).fit_predict(X_arr)
            view = evaluate_simulation_results(X_arr, y_true_arr, labels, best_k=k)
            score_value = float(getattr(view, score_name))
            leaderboard_rows.append(
                {
                    "k": k,
                    "seed": seed,
                    "accuracy": float(view.accuracy),
                    "balanced_accuracy": float(view.balanced_accuracy),
                    "cramers_v": float(view.cramers_v),
                }
            )

            if best is None or score_value > best.score_value:
                best = ClusteringSearchResult(
                    view=view,
                    k=k,
                    seed=seed,
                    score_name=score_name,
                    score_value=score_value,
                    feature_columns=[],
                    leaderboard=pd.DataFrame(),
                )

    assert best is not None
    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
        [score_name, "accuracy", "balanced_accuracy", "k", "seed"],
        ascending=[False, False, False, True, True],
    )
    return ClusteringSearchResult(
        view=best.view,
        k=best.k,
        seed=best.seed,
        score_name=best.score_name,
        score_value=best.score_value,
        feature_columns=[],
        leaderboard=leaderboard.reset_index(drop=True),
    )


def mode_or_first(x: pd.Series):
    """Return the modal value, falling back to the first non-null element."""
    x = x.dropna()
    if len(x) == 0:
        return pd.NA
    m = x.mode()
    return m.iloc[0] if len(m) else x.iloc[0]


def lonlat_to_xy_m(
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    lat0: float | None = None,
    lon0: float | None = None,
) -> np.ndarray:
    """Approximate lon/lat coordinates in meters with a local equirectangular projection."""
    radius_m = 6_371_000.0

    lat_arr = np.asarray(lat, dtype=float)
    lon_arr = np.asarray(lon, dtype=float)

    if lat0 is None:
        lat0 = float(np.nanmean(lat_arr))
    if lon0 is None:
        lon0 = float(np.nanmean(lon_arr))

    x = np.radians(lon_arr - lon0) * radius_m * np.cos(np.radians(lat0))
    y = np.radians(lat_arr - lat0) * radius_m
    return np.column_stack([x, y])


def get_xy_dataframe(
    df: pd.DataFrame,
    *,
    traj_col: str = DEFAULT_ID_COLUMN,
    label_col: str = DEFAULT_LABEL_COLUMN,
) -> pd.DataFrame:
    """Extract one point row per observation with planar x/y coordinates."""
    d = df.copy()
    d["__row_order"] = np.arange(len(d))

    if "geometry" in d.columns:
        geom = d["geometry"].dropna()
        if len(geom) > 0:
            first_geom = geom.iloc[0]
            if hasattr(first_geom, "x") and hasattr(first_geom, "y"):
                crs = getattr(d, "crs", None)
                if crs is not None:
                    try:
                        if d.crs.is_geographic:
                            try:
                                projected = d.to_crs(d.estimate_utm_crs())
                            except Exception:
                                projected = d.to_crs("EPSG:3857")
                            out = projected[[traj_col, label_col, "__row_order"]].copy()
                            out["x"] = projected.geometry.x
                            out["y"] = projected.geometry.y
                            return pd.DataFrame(out)

                        out = d[[traj_col, label_col, "__row_order"]].copy()
                        out["x"] = d.geometry.x
                        out["y"] = d.geometry.y
                        return pd.DataFrame(out)
                    except Exception:
                        pass

                raw_x = d["geometry"].apply(lambda p: p.x if p is not None else np.nan).to_numpy()
                raw_y = d["geometry"].apply(lambda p: p.y if p is not None else np.nan).to_numpy()
                looks_like_lonlat = (
                    np.nanmin(raw_x) >= -180
                    and np.nanmax(raw_x) <= 180
                    and np.nanmin(raw_y) >= -90
                    and np.nanmax(raw_y) <= 90
                )

                out = d[[traj_col, label_col, "__row_order"]].copy()
                if looks_like_lonlat:
                    xy = lonlat_to_xy_m(lat=raw_y, lon=raw_x)
                    out["x"] = xy[:, 0]
                    out["y"] = xy[:, 1]
                else:
                    out["x"] = raw_x
                    out["y"] = raw_y
                return out

    lat_candidates = ["latitude", "lat", "Latitude", "LAT"]
    lon_candidates = ["longitude", "lon", "lng", "Longitude", "LON"]
    lat_col = next((c for c in lat_candidates if c in d.columns), None)
    lon_col = next((c for c in lon_candidates if c in d.columns), None)

    if lat_col is not None and lon_col is not None:
        xy = lonlat_to_xy_m(
            lat=d[lat_col].to_numpy(dtype=float),
            lon=d[lon_col].to_numpy(dtype=float),
        )
        out = d[[traj_col, label_col, "__row_order"]].copy()
        out["x"] = xy[:, 0]
        out["y"] = xy[:, 1]
        return out

    if "x" in d.columns and "y" in d.columns:
        return d[[traj_col, label_col, "__row_order", "x", "y"]].copy()

    raise ValueError(
        "Could not find usable coordinates. Expected geometry, latitude/longitude, or x/y columns."
    )


def resample_trajectory(points: np.ndarray, *, n_points: int = 50) -> np.ndarray:
    """Arc-length resample a trajectory to a fixed number of points."""
    points_arr = np.asarray(points, dtype=float)
    if len(points_arr) == 0:
        return points_arr
    if len(points_arr) == 1:
        return np.repeat(points_arr, n_points, axis=0)

    step_dist = np.linalg.norm(np.diff(points_arr, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(step_dist)])

    keep = np.concatenate([[True], np.diff(arc) > 0])
    points_arr = points_arr[keep]
    arc = arc[keep]

    if len(points_arr) == 1 or arc[-1] == 0:
        return np.repeat(points_arr[:1], n_points, axis=0)

    arc_new = np.linspace(0, arc[-1], n_points)
    x_new = np.interp(arc_new, arc, points_arr[:, 0])
    y_new = np.interp(arc_new, arc, points_arr[:, 1])
    return np.column_stack([x_new, y_new])


def build_resampled_trajectory_list(
    df_xy: pd.DataFrame,
    *,
    order_col: str | None = None,
    n_resample: int = 50,
    traj_col: str = DEFAULT_ID_COLUMN,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Build one fixed-length resampled point sequence per trajectory id."""
    d = df_xy.copy()
    sort_cols = [traj_col, order_col] if order_col and order_col in d.columns else [traj_col, "__row_order"]

    traj_ids: list[str] = []
    trajectories: list[np.ndarray] = []

    for tid, group in d.sort_values(sort_cols).groupby(traj_col, sort=False):
        points = group[["x", "y"]].dropna().to_numpy(dtype=float)
        if len(points) < 2:
            continue
        traj_ids.append(str(tid))
        trajectories.append(resample_trajectory(points, n_points=n_resample))

    return np.asarray(traj_ids, dtype=object), trajectories


def discrete_frechet(P: np.ndarray, Q: np.ndarray) -> float:
    """Compute Eiter-Mannila discrete Frechet distance between two trajectories."""
    P_arr = np.asarray(P, dtype=float)
    Q_arr = np.asarray(Q, dtype=float)

    n = len(P_arr)
    m = len(Q_arr)
    D = np.linalg.norm(P_arr[:, None, :] - Q_arr[None, :, :], axis=2)
    ca = np.empty((n, m), dtype=float)

    ca[0, 0] = D[0, 0]
    for i in range(1, n):
        ca[i, 0] = max(ca[i - 1, 0], D[i, 0])
    for j in range(1, m):
        ca[0, j] = max(ca[0, j - 1], D[0, j])
    for i in range(1, n):
        for j in range(1, m):
            ca[i, j] = max(min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]), D[i, j])
    return float(ca[-1, -1])


def build_frechet_distance_matrix(trajectories: Sequence[np.ndarray]) -> np.ndarray:
    """Compute the full pairwise discrete-Frechet distance matrix."""
    n = len(trajectories)
    D = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            dist = discrete_frechet(trajectories[i], trajectories[j])
            D[i, j] = dist
            D[j, i] = dist
    return D


def _majority_mapped_accuracies(
    y_true: np.ndarray,
    cluster_labels: np.ndarray,
) -> tuple[float, float]:
    """Compute majority-vote mapped accuracy with and without noise points."""
    eval_df = pd.DataFrame(
        {
            "reference_lab": np.asarray(y_true, dtype=object),
            "cluster": np.asarray(cluster_labels),
        }
    )
    cluster_to_ref = eval_df.groupby("cluster")["reference_lab"].agg(
        lambda x: x.value_counts().idxmax()
    )
    mapped = eval_df["cluster"].map(cluster_to_ref)
    mapped_accuracy_all = float((mapped.astype(str) == eval_df["reference_lab"].astype(str)).mean())

    non_noise = eval_df["cluster"] != -1
    if non_noise.any():
        mapped_accuracy_non_noise = float(
            (
                mapped.loc[non_noise].astype(str)
                == eval_df.loc[non_noise, "reference_lab"].astype(str)
            ).mean()
        )
    else:
        mapped_accuracy_non_noise = float("nan")

    return mapped_accuracy_all, mapped_accuracy_non_noise


def search_best_frechet_dbscan(
    points: pd.DataFrame,
    *,
    label_col: str = DEFAULT_LABEL_COLUMN,
    id_col: str = DEFAULT_ID_COLUMN,
    order_col: str | None = None,
    n_resample: int = 50,
    min_samples: int = 5,
    eps_percentiles: Sequence[int] = (50, 60, 70, 75, 80, 85, 90, 95),
    score_name: str = "accuracy",
) -> FrechetDbscanResult:
    """Run Frechet+DBSCAN across eps percentiles and return the best baseline run."""
    if score_name not in {"accuracy", "balanced_accuracy"}:
        raise ValueError("score_name must be 'accuracy' or 'balanced_accuracy'.")

    df_xy = get_xy_dataframe(points, traj_col=id_col, label_col=label_col)
    if order_col and order_col in points.columns:
        df_xy[order_col] = points[order_col].values

    traj_ids, trajectories = build_resampled_trajectory_list(
        df_xy,
        order_col=order_col,
        n_resample=n_resample,
        traj_col=id_col,
    )
    if len(trajectories) == 0:
        raise ValueError("No usable trajectories were found for Frechet clustering.")

    traj_true = (
        points.groupby(id_col)[label_col]
        .agg(mode_or_first)
        .reset_index()
    )
    traj_true = traj_true[traj_true[id_col].astype(str).isin(traj_ids)].copy()
    traj_true = traj_true.set_index(traj_true[id_col].astype(str)).loc[traj_ids]
    y_true = traj_true[label_col].to_numpy(dtype=object)

    D = build_frechet_distance_matrix(trajectories)
    D_sorted = np.sort(D, axis=1)
    k_index = max(0, min(min_samples - 1, D_sorted.shape[1] - 1))
    kdist = D_sorted[:, k_index]

    best: FrechetDbscanResult | None = None
    leaderboard_rows: list[dict[str, float | int]] = []

    for q in eps_percentiles:
        eps = float(np.percentile(kdist, q))
        labels = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed").fit_predict(D)
        view = evaluate_simulation_results(D, y_true, labels, best_k=None)
        ari = float(adjusted_rand_score(y_true.astype(str), labels.astype(str)))
        nmi = float(normalized_mutual_info_score(y_true.astype(str), labels.astype(str)))
        mapped_accuracy_all, mapped_accuracy_non_noise = _majority_mapped_accuracies(y_true, labels)
        n_clusters = int(len(set(labels)) - int(-1 in labels))
        noise_fraction = float(np.mean(labels == -1))
        score_value = float(getattr(view, score_name))

        leaderboard_rows.append(
            {
                "eps_percentile": int(q),
                "eps": eps,
                "accuracy": float(view.accuracy),
                "balanced_accuracy": float(view.balanced_accuracy),
                "ari": ari,
                "nmi": nmi,
                "mapped_accuracy_all": mapped_accuracy_all,
                "mapped_accuracy_non_noise": mapped_accuracy_non_noise,
                "n_clusters": n_clusters,
                "noise_fraction": noise_fraction,
            }
        )

        if best is None or score_value > best.score_value:
            best = FrechetDbscanResult(
                view=view,
                eps_percentile=int(q),
                eps=eps,
                min_samples=min_samples,
                score_name=score_name,
                score_value=score_value,
                ari=ari,
                nmi=nmi,
                mapped_accuracy_all=mapped_accuracy_all,
                mapped_accuracy_non_noise=mapped_accuracy_non_noise,
                n_clusters=n_clusters,
                noise_fraction=noise_fraction,
                leaderboard=pd.DataFrame(),
            )

    assert best is not None
    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
        [score_name, "accuracy", "balanced_accuracy", "ari", "eps_percentile"],
        ascending=[False, False, False, False, True],
    )
    return FrechetDbscanResult(
        view=best.view,
        eps_percentile=best.eps_percentile,
        eps=best.eps,
        min_samples=best.min_samples,
        score_name=best.score_name,
        score_value=best.score_value,
        ari=best.ari,
        nmi=best.nmi,
        mapped_accuracy_all=best.mapped_accuracy_all,
        mapped_accuracy_non_noise=best.mapped_accuracy_non_noise,
        n_clusters=best.n_clusters,
        noise_fraction=best.noise_fraction,
        leaderboard=leaderboard.reset_index(drop=True),
    )
