#!/usr/bin/env python3
"""Downsample simulation CSV/parquet inputs to a requested label split."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd

from evacmob.simulation_subset import (
    SimulationSubsetConfig,
    subset_simulation_points,
    subset_simulation_trajectories,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Downsample simulation inputs by reference label.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("hourly_locations_wide_300x143_plus_reference_lab.csv"),
        help="Trajectory-level CSV with a reference label column.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("hourly_locations_wide_25_25_75.csv"),
        help="Destination for the filtered trajectory CSV.",
    )
    parser.add_argument(
        "--input-parquet",
        type=Path,
        default=Path("ref_simulation_point_gdf.parquet"),
        help="Optional point-level parquet keyed by traj_id.",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=Path("ref_simulation_point_gdf_25_25_75.parquet"),
        help="Destination for the filtered point parquet.",
    )
    parser.add_argument("--label-column", default="reference_lab")
    parser.add_argument("--id-column", default="traj_id")
    parser.add_argument("--compact-local", type=int, default=25)
    parser.add_argument("--intermediate-directed", type=int, default=25)
    parser.add_argument("--extensive-displacement", type=int, default=75)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    trajectories = pd.read_csv(args.input_csv)
    targets = SimulationSubsetConfig(
        compact_local=args.compact_local,
        intermediate_directed=args.intermediate_directed,
        extensive_displacement=args.extensive_displacement,
    )
    subset = subset_simulation_trajectories(
        trajectories,
        label_col=args.label_column,
        id_col=args.id_column,
        target_counts=targets.as_mapping(),
        seed=args.seed,
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(subset)} trajectory rows to {args.output_csv}")
    print(subset[args.label_column].value_counts().sort_index().to_string())

    if args.input_parquet.exists():
        points = gpd.read_parquet(args.input_parquet)
        points_subset = subset_simulation_points(
            points,
            subset[args.id_column].tolist(),
            id_col=args.id_column,
        )
        args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
        points_subset.to_parquet(args.output_parquet, index=False)
        print(f"Wrote {len(points_subset)} point rows to {args.output_parquet}")
    else:
        print(f"Skipped parquet filtering; {args.input_parquet} does not exist.")


if __name__ == "__main__":
    main()
