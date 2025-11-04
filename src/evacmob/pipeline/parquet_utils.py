"""Helpers for working with Parquet datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import pyarrow.parquet as pq


def read_parquet_row_groups(
    path: Path | str,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Read all row groups from a Parquet file into a pandas DataFrame.

    Mirrors the row-group aware loader used in the notebook pipeline to avoid
    eager loading of unneeded columns.
    """
    parquet_file = pq.ParquetFile(path)
    frames = [
        parquet_file.read_row_group(i, columns=columns).to_pandas()
        for i in range(parquet_file.num_row_groups)
    ]
    return pd.concat(frames, ignore_index=True)
