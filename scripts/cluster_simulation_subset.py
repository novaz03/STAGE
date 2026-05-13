#!/usr/bin/env python3
"""Run repeated KMeans on a simulation subset and report the best aligned score."""

from __future__ import annotations

import argparse
from pathlib import Path

from evacmob.simulation_clustering import load_clustering_input, search_best_kmeans
from evacmob.simulation_results import format_simulation_results


def _parse_k_values(raw: str) -> list[int]:
    values = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    if not values:
        raise ValueError("At least one k value must be provided.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster a simulation subset and score it.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("hourly_locations_wide_25_25_75.csv"),
        help="Trajectory CSV or point parquet to cluster.",
    )
    parser.add_argument("--label-column", default="reference_lab")
    parser.add_argument("--id-column", default="traj_id")
    parser.add_argument("--k-values", default="2,3,4,5,6")
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument(
        "--score",
        choices=["accuracy", "balanced_accuracy"],
        default="accuracy",
        help="Metric used to choose the best clustering run.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    parser.add_argument("--topn", type=int, default=10, help="Number of leaderboard rows to print.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    k_values = _parse_k_values(args.k_values)
    seeds = list(range(args.n_seeds))

    feature_table, X, y_true, feature_columns, _ = load_clustering_input(
        args.input,
        label_col=args.label_column,
        id_col=args.id_column,
    )
    result = search_best_kmeans(
        X,
        y_true,
        k_values=k_values,
        seeds=seeds,
        score_name=args.score,
        show_progress=not args.no_progress,
    )

    print(f"Input: {args.input}")
    print(f"Rows clustered: {X.shape[0]}")
    print(f"Feature columns used ({len(feature_columns)}):")
    print(", ".join(feature_columns))
    print()
    print(f"Best {result.score_name}: {result.score_value:.4f}")
    print(f"Best run: k={result.k}, seed={result.seed}")
    print()
    print(format_simulation_results(result.view))
    print("=== Top runs ===")
    print(result.leaderboard.head(args.topn).to_string(index=False))


if __name__ == "__main__":
    main()
