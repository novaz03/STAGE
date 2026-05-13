import numpy as np
import pandas as pd

from evacmob.simulation_clustering import prepare_point_feature_table, search_best_kmeans


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
