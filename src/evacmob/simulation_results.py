"""Simulation-result viewing helpers for clustering diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2_contingency
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report


@dataclass(frozen=True)
class SimulationResultsView:
    """Container for the full set of clustering-vs-truth diagnostics."""

    best_k: int | None
    cluster_sizes: pd.Series
    cluster_to_truth_map: dict[Any, Any]
    raw_contingency: pd.DataFrame
    chi2: float
    dof: int
    p_value: float
    cramers_v: float
    aligned_confusion_matrix: pd.DataFrame
    accuracy: float
    balanced_accuracy: float
    classification_report_text: str
    aligned_predictions: np.ndarray


def align_clusters_to_truth(
    y_true: np.ndarray | pd.Series | list[Any],
    cluster_labels: np.ndarray | pd.Series | list[Any],
) -> tuple[np.ndarray, dict[Any, Any], pd.DataFrame]:
    """Align cluster IDs to truth labels using Hungarian assignment."""
    y_true_arr = np.asarray(y_true)
    cluster_arr = np.asarray(cluster_labels)

    true_ids = list(pd.unique(y_true_arr))
    cluster_ids = sorted(np.unique(cluster_arr))

    contingency = pd.crosstab(
        pd.Categorical(y_true_arr, categories=true_ids, ordered=True),
        pd.Categorical(cluster_arr, categories=cluster_ids, ordered=True),
        dropna=False,
    )
    counts = contingency.to_numpy()

    row_ind, col_ind = linear_sum_assignment(-counts.astype(float))

    mapping: dict[Any, Any] = {}
    assigned_clusters = set()
    for i, j in zip(row_ind, col_ind):
        mapping[cluster_ids[j]] = true_ids[i]
        assigned_clusters.add(cluster_ids[j])

    # Fallback when number of clusters differs from number of truth classes.
    for cluster_id in cluster_ids:
        if cluster_id in assigned_clusters:
            continue
        j = cluster_ids.index(cluster_id)
        i_best = int(np.argmax(counts[:, j]))
        mapping[cluster_id] = true_ids[i_best]

    aligned_pred = np.array([mapping[label] for label in cluster_arr], dtype=object)
    return aligned_pred, mapping, contingency


def evaluate_simulation_results(
    X: np.ndarray | pd.DataFrame,
    y_true: np.ndarray | pd.Series | list[Any],
    final_labels: np.ndarray | pd.Series | list[Any],
    best_k: int | None = None,
) -> SimulationResultsView:
    """Compute contingency/chi-square and aligned confusion metrics for a simulation run."""
    y_true_arr = np.asarray(y_true)
    labels_arr = np.asarray(final_labels)
    n_samples = X.shape[0]

    if not (len(y_true_arr) == len(labels_arr) == n_samples):
        raise ValueError("Length mismatch among y_true, final_labels, and X.")

    cluster_sizes = pd.Series(labels_arr, name="cluster").value_counts().sort_index()

    aligned_pred, cluster_to_truth_map, _ = align_clusters_to_truth(
        y_true=y_true_arr,
        cluster_labels=labels_arr,
    )

    contingency = pd.crosstab(
        pd.Series(y_true_arr, name="ground_truth"),
        pd.Series(labels_arr, name="cluster"),
        dropna=False,
    )

    chi2, p_value, dof, _ = chi2_contingency(contingency)
    n = contingency.to_numpy().sum()
    r, c = contingency.shape
    cramers_v = np.sqrt((chi2 / n) / max(1, min(r - 1, c - 1)))

    aligned_confusion = pd.crosstab(
        pd.Series(y_true_arr, name="ground_truth"),
        pd.Series(aligned_pred, name="pred_from_cluster"),
        dropna=False,
    )

    accuracy = accuracy_score(y_true_arr, aligned_pred)
    balanced_accuracy = balanced_accuracy_score(y_true_arr, aligned_pred)
    report_text = classification_report(y_true_arr, aligned_pred, zero_division=0)

    return SimulationResultsView(
        best_k=best_k,
        cluster_sizes=cluster_sizes,
        cluster_to_truth_map=cluster_to_truth_map,
        raw_contingency=contingency,
        chi2=float(chi2),
        dof=int(dof),
        p_value=float(p_value),
        cramers_v=float(cramers_v),
        aligned_confusion_matrix=aligned_confusion,
        accuracy=float(accuracy),
        balanced_accuracy=float(balanced_accuracy),
        classification_report_text=report_text,
        aligned_predictions=aligned_pred,
    )


def format_simulation_results(view: SimulationResultsView) -> str:
    """Format a ``SimulationResultsView`` similar to notebook print output."""
    lines: list[str] = []
    if view.best_k is not None:
        lines.append(f"Using best_k = {view.best_k}")
    lines.append("Cluster sizes:")
    lines.append(str(view.cluster_sizes))

    lines.append("\nCluster -> truth mapping used for aligned confusion matrix:")
    lines.append(str(view.cluster_to_truth_map))

    lines.append("\n=== Raw contingency table: ground truth vs cluster ===")
    lines.append(str(view.raw_contingency))

    lines.append("\n=== Chi-square test ===")
    lines.append(f"chi2      = {view.chi2:.4f}")
    lines.append(f"dof       = {view.dof}")
    lines.append(f"p_value   = {view.p_value:.6g}")
    lines.append(f"Cramer's V = {view.cramers_v:.4f}")

    lines.append("\n=== Aligned confusion matrix ===")
    lines.append(str(view.aligned_confusion_matrix))

    lines.append("\n=== Accuracy metrics after alignment ===")
    lines.append(f"Accuracy          = {view.accuracy:.4f}")
    lines.append(f"Balanced accuracy = {view.balanced_accuracy:.4f}")

    lines.append("\n=== Classification report ===")
    lines.append(view.classification_report_text.rstrip())

    return "\n".join(lines) + "\n"


def print_simulation_results(view: SimulationResultsView) -> None:
    """Print a formatted simulation-result summary."""
    print(format_simulation_results(view))


def view_simulation_results(
    X: np.ndarray | pd.DataFrame,
    y_true: np.ndarray | pd.Series | list[Any],
    final_labels: np.ndarray | pd.Series | list[Any],
    *,
    best_k: int | None = None,
) -> SimulationResultsView:
    """One-shot helper: evaluate and print simulation results."""
    view = evaluate_simulation_results(X=X, y_true=y_true, final_labels=final_labels, best_k=best_k)
    print_simulation_results(view)
    return view


def view_simulation_results_from_stability(
    X: np.ndarray | pd.DataFrame,
    y_true: np.ndarray | pd.Series | list[Any],
    stability_result: Mapping[str, Any],
    *,
    labels_key: str = "consensus_labels",
    best_k: int | None = None,
) -> SimulationResultsView:
    """Convenience wrapper that reads labels from a stability-result mapping."""
    if labels_key not in stability_result:
        raise KeyError(f"'{labels_key}' not found in stability_result.")
    final_labels = np.asarray(stability_result[labels_key])
    return view_simulation_results(X=X, y_true=y_true, final_labels=final_labels, best_k=best_k)
