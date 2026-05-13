#!/usr/bin/env python3
"""Compare engineered-feature KMeans against a Frechet+DBSCAN baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd

from evacmob.simulation_clustering import (
    load_clustering_input,
    search_best_frechet_dbscan,
    search_best_kmeans,
)
from evacmob.simulation_results import format_simulation_results


def _parse_k_values(raw: str) -> list[int]:
    values = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    if not values:
        raise ValueError("At least one k value must be provided.")
    return values


def _parse_percentiles(raw: str) -> list[int]:
    values = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    if not values:
        raise ValueError("At least one eps percentile must be provided.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare KMeans clustering with a Frechet+DBSCAN baseline."
    )
    parser.add_argument(
        "--input-parquet",
        type=Path,
        default=Path("ref_simulation_point_gdf_25_25_75.parquet"),
        help="Point-level parquet used by both methods.",
    )
    parser.add_argument("--label-column", default="reference_lab")
    parser.add_argument("--id-column", default="traj_id")
    parser.add_argument("--order-column", default=None)
    parser.add_argument("--k-values", default="2,3,4,5,6")
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--score", choices=["accuracy", "balanced_accuracy"], default="accuracy")
    parser.add_argument("--n-resample", type=int, default=50)
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--eps-percentiles", default="50,60,70,75,80,85,90,95")
    parser.add_argument("--topn", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    k_values = _parse_k_values(args.k_values)
    seeds = list(range(args.n_seeds))
    eps_percentiles = _parse_percentiles(args.eps_percentiles)

    points = gpd.read_parquet(args.input_parquet)

    _, X, y_true, feature_columns, _ = load_clustering_input(
        args.input_parquet,
        label_col=args.label_column,
        id_col=args.id_column,
    )
    kmeans_result = search_best_kmeans(
        X,
        y_true,
        k_values=k_values,
        seeds=seeds,
        score_name=args.score,
    )

    frechet_result = search_best_frechet_dbscan(
        points,
        label_col=args.label_column,
        id_col=args.id_column,
        order_col=args.order_column,
        n_resample=args.n_resample,
        min_samples=args.min_samples,
        eps_percentiles=eps_percentiles,
        score_name=args.score,
    )

    summary = [
        {
            "method": "kmeans",
            "score": kmeans_result.score_value,
            "accuracy": kmeans_result.view.accuracy,
            "balanced_accuracy": kmeans_result.view.balanced_accuracy,
            "cramers_v": kmeans_result.view.cramers_v,
            "config": f"k={kmeans_result.k}, seed={kmeans_result.seed}",
        },
        {
            "method": "frechet_dbscan",
            "score": frechet_result.score_value,
            "accuracy": frechet_result.view.accuracy,
            "balanced_accuracy": frechet_result.view.balanced_accuracy,
            "cramers_v": frechet_result.view.cramers_v,
            "config": (
                f"eps_pct={frechet_result.eps_percentile}, "
                f"eps={frechet_result.eps:.3f}, min_samples={frechet_result.min_samples}"
            ),
        },
    ]

    print(f"Input parquet: {args.input_parquet}")
    print(f"Trajectory feature columns for KMeans ({len(feature_columns)}):")
    print(", ".join(feature_columns))
    print()
    print("=== Method Summary ===")
    for row in summary:
        print(
            f"{row['method']}: score={row['score']:.4f}, "
            f"accuracy={row['accuracy']:.4f}, "
            f"balanced_accuracy={row['balanced_accuracy']:.4f}, "
            f"cramers_v={row['cramers_v']:.4f}, {row['config']}"
        )

    print()
    print("=== Best KMeans Result ===")
    print(format_simulation_results(kmeans_result.view))
    print("=== Top KMeans Runs ===")
    print(kmeans_result.leaderboard.head(args.topn).to_string(index=False))

    print()
    print("=== Best Frechet+DBSCAN Result ===")
    print(
        f"ARI={frechet_result.ari:.4f}, NMI={frechet_result.nmi:.4f}, "
        f"mapped_accuracy_all={frechet_result.mapped_accuracy_all:.4f}, "
        f"mapped_accuracy_non_noise={frechet_result.mapped_accuracy_non_noise:.4f}, "
        f"n_clusters={frechet_result.n_clusters}, "
        f"noise_fraction={frechet_result.noise_fraction:.4f}"
    )
    print(format_simulation_results(frechet_result.view))
    print("=== Top Frechet+DBSCAN Runs ===")
    print(frechet_result.leaderboard.head(args.topn).to_string(index=False))


if __name__ == "__main__":
    main()
