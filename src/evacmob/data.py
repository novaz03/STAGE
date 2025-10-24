"""Data loading/saving utilities."""
from pathlib import Path
import pandas as pd

def load_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)

def save_parquet(df, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
