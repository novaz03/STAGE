"""Helpers to downsample simulation inputs while preserving label counts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import pandas as pd

from .simulation_label_assignment import (
    LABEL_COMPACT_LOCAL,
    LABEL_EXTENSIVE_DISPLACEMENT,
    LABEL_INTERMEDIATE_DIRECTED,
)


@dataclass(frozen=True)
class SimulationSubsetConfig:
    """Target counts for the three semantic simulation labels."""

    compact_local: int = 25
    intermediate_directed: int = 25
    extensive_displacement: int = 75

    def as_mapping(self) -> dict[str, int]:
        return {
            LABEL_COMPACT_LOCAL: self.compact_local,
            LABEL_INTERMEDIATE_DIRECTED: self.intermediate_directed,
            LABEL_EXTENSIVE_DISPLACEMENT: self.extensive_displacement,
        }


def subset_simulation_trajectories(
    trajectories: pd.DataFrame,
    *,
    label_col: str = "reference_lab",
    id_col: str = "traj_id",
    target_counts: Mapping[str, int] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Downsample trajectories to requested counts per semantic label."""
    if label_col not in trajectories.columns:
        raise KeyError(f"'{label_col}' not found in trajectories.")
    if id_col not in trajectories.columns:
        raise KeyError(f"'{id_col}' not found in trajectories.")

    targets = dict(target_counts or SimulationSubsetConfig().as_mapping())
    missing_labels = [label for label in targets if label not in set(trajectories[label_col])]
    if missing_labels:
        raise ValueError(f"Requested labels missing from input: {missing_labels}")

    selected_indices: list[int] = []
    for offset, (label, count) in enumerate(targets.items()):
        if count < 0:
            raise ValueError(f"Requested count for '{label}' must be non-negative.")

        label_rows = trajectories[trajectories[label_col] == label]
        available = len(label_rows)
        if available < count:
            raise ValueError(
                f"Requested {count} rows for '{label}' but only {available} are available."
            )

        sampled = label_rows.sample(n=count, random_state=seed + offset, replace=False)
        selected_indices.extend(sampled.index.tolist())

    subset = trajectories.loc[sorted(selected_indices)].copy()
    subset = subset.reset_index(drop=True)
    return subset


def subset_simulation_points(
    points: pd.DataFrame,
    trajectory_ids: Iterable[str],
    *,
    id_col: str = "traj_id",
) -> pd.DataFrame:
    """Filter point-level records to the selected trajectory ids."""
    if id_col not in points.columns:
        raise KeyError(f"'{id_col}' not found in points.")

    ids = {str(traj_id) for traj_id in trajectory_ids}
    mask = points[id_col].astype(str).isin(ids)
    return points.loc[mask].copy()
