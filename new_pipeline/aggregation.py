"""Aggregation helpers for spatial embeddings."""

from __future__ import annotations

import logging
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def mean_vectors_by_group(
    df: pd.DataFrame,
    group_column: str,
    vector_column: str,
    count_column: str = "poi_count",
) -> pd.DataFrame:
    """
    Compute the mean vector for each group.

    Parameters
    ----------
    df : DataFrame
        Must contain ``group_column`` and ``vector_column`` where vectors are
        sequences convertible to numpy arrays.
    group_column : str
        Column name denoting the group ID (e.g., hexagon or block group).
    vector_column : str
        Column containing the vector representations (list/array-like).
    count_column : str
        Name of the column that stores the number of contributing items.

    Returns
    -------
    DataFrame with columns ``group_column``, ``vector_column`` and
    ``count_column`` where the vectors are averaged.
    """
    if group_column not in df.columns:
        raise ValueError(f"Missing group column '{group_column}'")
    if vector_column not in df.columns:
        raise ValueError(f"Missing vector column '{vector_column}'")

    valid = df.dropna(subset=[group_column, vector_column])
    if valid.empty:
        LOGGER.warning("No valid rows available for aggregation.")
        return pd.DataFrame(columns=[group_column, vector_column, count_column])

    def _mean_stack(values: Iterable) -> np.ndarray:
        vectors = [np.asarray(v, dtype=np.float32) for v in values]
        return np.stack(vectors).mean(axis=0)

    aggregated = valid.groupby(group_column)[vector_column].agg(_mean_stack).reset_index()
    counts = valid.groupby(group_column)[vector_column].size().rename(count_column).reset_index()
    return aggregated.merge(counts, on=group_column, how="left")
