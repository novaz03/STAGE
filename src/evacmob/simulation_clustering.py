"""Clustering helpers for labeled simulation subsets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
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
