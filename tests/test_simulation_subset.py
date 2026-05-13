import pandas as pd

from evacmob.simulation_subset import (
    SimulationSubsetConfig,
    subset_simulation_points,
    subset_simulation_trajectories,
)


def _build_trajectory_df() -> pd.DataFrame:
    rows = []
    labels = [
        ("compact_local", 4),
        ("intermediate_directed", 4),
        ("extensive_displacement", 5),
    ]
    for label, count in labels:
        for idx in range(count):
            rows.append(
                {
                    "traj_id": f"{label[:3]}-{idx}",
                    "reference_lab": label,
                    "feature": idx,
                }
            )
    return pd.DataFrame(rows)


def test_subset_simulation_trajectories_obeys_requested_counts():
    trajectories = _build_trajectory_df()
    targets = SimulationSubsetConfig(
        compact_local=2,
        intermediate_directed=1,
        extensive_displacement=3,
    )

    subset = subset_simulation_trajectories(
        trajectories,
        target_counts=targets.as_mapping(),
        seed=7,
    )

    assert subset["reference_lab"].value_counts().sort_index().to_dict() == {
        "compact_local": 2,
        "extensive_displacement": 3,
        "intermediate_directed": 1,
    }
    assert subset["traj_id"].is_unique


def test_subset_simulation_points_follows_selected_trajectory_ids():
    points = pd.DataFrame(
        {
            "traj_id": ["com-0", "com-0", "int-0", "ext-0", "keep-me"],
            "pt_idx": [1, 2, 1, 1, 1],
        }
    )

    subset = subset_simulation_points(points, ["com-0", "keep-me"])

    assert subset["traj_id"].tolist() == ["com-0", "com-0", "keep-me"]


def test_subset_simulation_trajectories_raises_when_request_exceeds_available():
    trajectories = _build_trajectory_df()

    try:
        subset_simulation_trajectories(
            trajectories,
            target_counts={"compact_local": 10},
        )
    except ValueError as exc:
        assert "only 4 are available" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError when requesting too many rows.")
