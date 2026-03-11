"""Simulation label reassignment using a saved classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LABEL_COMPACT_LOCAL = "compact_local"
LABEL_INTERMEDIATE_DIRECTED = "intermediate_directed"
LABEL_EXTENSIVE_DISPLACEMENT = "extensive_displacement"

RAW_TO_SEMANTIC_LABEL = {
    "sip_home_grocery": LABEL_COMPACT_LOCAL,
    "sip_hospital": LABEL_INTERMEDIATE_DIRECTED,
    "evac_out_of_zone": LABEL_EXTENSIVE_DISPLACEMENT,
}


def _load_model(model_path: str | Path) -> Any:
    try:
        import joblib
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "joblib is required to load the reassignment model. Install scikit-learn/joblib."
        ) from exc
    return joblib.load(model_path)


def _get_feature_names(model: Any) -> list[str]:
    if hasattr(model, "feature_names_in_"):
        return [str(c) for c in model.feature_names_in_]
    if hasattr(model, "named_steps"):
        for step in reversed(list(model.named_steps.values())):
            if hasattr(step, "feature_names_in_"):
                return [str(c) for c in step.feature_names_in_]
    raise ValueError(
        "Could not determine model input schema. Model must expose `feature_names_in_`."
    )


def prepare_reassignment_features(features_df: pd.DataFrame, model: Any) -> pd.DataFrame:
    """Build a model-aligned numeric feature frame for reassignment."""
    expected_cols = _get_feature_names(model)
    aligned = features_df.reindex(columns=expected_cols, fill_value=np.nan).copy()

    for col in aligned.columns:
        if pd.api.types.is_bool_dtype(aligned[col]):
            aligned[col] = aligned[col].astype(int)

    aligned = aligned.apply(pd.to_numeric, errors="coerce")
    aligned = aligned.replace([np.inf, -np.inf], np.nan)
    return aligned


def reassign_simulation_labels(
    features_df: pd.DataFrame,
    *,
    model: Any | None = None,
    model_path: str | Path = "knn_k3.joblib",
    id_col: str = "traj_id",
    output_col: str = "traj_cluster",
    raw_pred_col: str = "knn_k3_label_raw",
) -> pd.DataFrame:
    """Predict and overwrite simulation labels with semantic reassigned labels.

    `features_df` should be a trajectory-level table containing the model's feature columns.
    """
    if model is None:
        model = _load_model(model_path)

    X = prepare_reassignment_features(features_df, model=model)
    pred_raw = np.asarray(model.predict(X), dtype=object)
    pred_semantic = (
        pd.Series(pred_raw, index=features_df.index)
        .map(RAW_TO_SEMANTIC_LABEL)
        .fillna(pd.Series(pred_raw, index=features_df.index))
        .to_numpy(dtype=object)
    )

    out = features_df.copy()
    out[raw_pred_col] = pred_raw
    out[output_col] = pred_semantic

    if id_col in out.columns:
        out = out.sort_values(id_col).reset_index(drop=True)
    return out


def build_reassigned_label_table(
    reassigned_df: pd.DataFrame,
    *,
    id_col: str = "traj_id",
    output_col: str = "traj_cluster",
) -> pd.DataFrame:
    """Return one row per trajectory id for downstream merges."""
    if id_col not in reassigned_df.columns:
        raise KeyError(f"'{id_col}' not found in reassigned_df.")
    if output_col not in reassigned_df.columns:
        raise KeyError(f"'{output_col}' not found in reassigned_df.")
    return reassigned_df[[id_col, output_col]].drop_duplicates(id_col).copy()


def merge_reassigned_labels(
    target_df: pd.DataFrame,
    reassigned_labels_df: pd.DataFrame,
    *,
    id_col: str = "traj_id",
    output_col: str = "traj_cluster",
) -> pd.DataFrame:
    """Attach reassigned labels to another table by trajectory id."""
    if id_col not in target_df.columns:
        raise KeyError(f"'{id_col}' not found in target_df.")
    if id_col not in reassigned_labels_df.columns:
        raise KeyError(f"'{id_col}' not found in reassigned_labels_df.")

    labels = reassigned_labels_df[[id_col, output_col]].drop_duplicates(id_col)
    merged = target_df.drop(columns=[output_col], errors="ignore").merge(
        labels,
        on=id_col,
        how="left",
        validate="many_to_one",
    )
    return merged


def save_reassigned_labels(df: pd.DataFrame, out_path: str | Path) -> Path:
    """Save reassigned labels table to CSV or Parquet based on extension."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
    elif suffix in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
    else:
        raise ValueError("Unsupported output format. Use .csv or .parquet.")
    return path
