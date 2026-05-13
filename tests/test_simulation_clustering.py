import numpy as np
import pandas as pd

from evacmob.simulation_clustering import (
    discrete_frechet,
    prepare_point_feature_table,
    search_best_frechet_dbscan,
    search_best_kmeans,
)


def test_prepare_point_feature_table_builds_one_row_per_trajectory():
    points = pd.DataFrame(
        {
            "traj_id": ["a", "a", "b", "b"],
            "pt_idx": [1, 2, 1, 2],
            "latitude": [0.0, 1.0, 10.0, 11.0],
            "longitude": [0.0, 1.0, 10.0, 11.0],
            "reference_lab": ["compact_local", "compact_local", "extensive_displacement", "extensive_displacement"],
        }
    )

    feature_table, X, y_true, feature_columns, traj_ids = prepare_point_feature_table(points)

    assert list(traj_ids) == ["a", "b"]
    assert feature_table.shape[0] == 2
    assert X.shape == (2, len(feature_columns))
    assert list(y_true) == ["compact_local", "extensive_displacement"]
    assert "net_disp_deg" in feature_columns


def test_search_best_kmeans_finds_perfect_split_on_easy_data():
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.2],
            [10.0, 10.0],
            [10.2, 10.1],
            [20.0, 20.0],
            [20.1, 20.2],
        ],
        dtype=float,
    )
    y_true = np.array(
        [
            "compact_local",
            "compact_local",
            "intermediate_directed",
            "intermediate_directed",
            "extensive_displacement",
            "extensive_displacement",
        ],
        dtype=object,
    )

    result = search_best_kmeans(
        X,
        y_true,
        k_values=[3],
        seeds=[0, 1, 2],
        score_name="accuracy",
    )

    assert result.k == 3
    assert np.isclose(result.score_value, 1.0)
    assert np.isclose(result.view.accuracy, 1.0)


def test_discrete_frechet_is_zero_for_identical_trajectories():
    traj = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 1.0]], dtype=float)
    assert np.isclose(discrete_frechet(traj, traj), 0.0)


def test_search_best_frechet_dbscan_finds_clean_split_on_easy_trajectories():
    points = pd.DataFrame(
        {
            "traj_id": [
                "a1",
                "a1",
                "a2",
                "a2",
                "b1",
                "b1",
                "b2",
                "b2",
            ],
            "pt_idx": [1, 2, 1, 2, 1, 2, 1, 2],
            "latitude": [0.0, 0.001, 0.0, 0.0011, 1.0, 1.001, 1.0, 1.0011],
            "longitude": [0.0, 0.001, 0.0001, 0.0012, 1.0, 1.001, 1.0001, 1.0012],
            "reference_lab": [
                "compact_local",
                "compact_local",
                "compact_local",
                "compact_local",
                "extensive_displacement",
                "extensive_displacement",
                "extensive_displacement",
                "extensive_displacement",
            ],
        }
    )

    result = search_best_frechet_dbscan(
        points,
        n_resample=8,
        min_samples=1,
        eps_percentiles=[10, 50, 90],
        score_name="accuracy",
    )

    assert np.isclose(result.view.accuracy, 1.0)
    assert np.isclose(result.score_value, 1.0)
    assert result.n_clusters >= 2
