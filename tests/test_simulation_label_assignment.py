import numpy as np
import pandas as pd

from evacmob.simulation_label_assignment import (
    RAW_TO_ASSIGNED_LABEL,
    assign_simulation_labels,
    build_assigned_label_table,
    merge_assigned_labels,
    prepare_label_assignment_features,
)


class _DummyModel:
    feature_names_in_ = np.array(["f1", "f2", "f_missing"])

    def predict(self, X):
        labels = []
        for val in X["f1"].fillna(0.0).to_numpy():
            if val > 0:
                labels.append("sip_home_grocery")
            elif val < 0:
                labels.append("sip_hospital")
            else:
                labels.append("evac_out_of_zone")
        return np.asarray(labels, dtype=object)


def test_prepare_label_assignment_features_aligns_schema():
    df = pd.DataFrame({"traj_id": ["a", "b"], "f2": [True, False], "f1": [1.0, -2.0]})
    X = prepare_label_assignment_features(df, _DummyModel())

    assert list(X.columns) == ["f1", "f2", "f_missing"]
    assert X["f2"].tolist() == [1, 0]
    assert X["f_missing"].isna().all()


def test_assign_simulation_labels_writes_semantic_traj_cluster():
    df = pd.DataFrame({"traj_id": ["b", "a", "c"], "f1": [1.0, -1.0, 0.0], "f2": [1, 0, 1]})
    out = assign_simulation_labels(df, model=_DummyModel())

    assert out["traj_id"].tolist() == ["a", "b", "c"]
    assert out["knn_k3_label_raw"].tolist() == [
        "sip_hospital",
        "sip_home_grocery",
        "evac_out_of_zone",
    ]
    assert out["traj_cluster"].tolist() == [
        RAW_TO_ASSIGNED_LABEL["sip_hospital"],
        RAW_TO_ASSIGNED_LABEL["sip_home_grocery"],
        RAW_TO_ASSIGNED_LABEL["evac_out_of_zone"],
    ]


def test_build_and_merge_assigned_labels():
    assigned = pd.DataFrame(
        {
            "traj_id": ["t1", "t2", "t3"],
            "traj_cluster": ["compact_local", "intermediate_directed", "extensive_displacement"],
        }
    )
    label_table = build_assigned_label_table(assigned)
    assert label_table.shape == (3, 2)

    points = pd.DataFrame({"traj_id": ["t1", "t1", "t3", "x"], "value": [1, 2, 3, 4]})
    merged = merge_assigned_labels(points, label_table)
    assert merged["traj_cluster"].iloc[0] == "compact_local"
    assert merged["traj_cluster"].iloc[1] == "compact_local"
    assert merged["traj_cluster"].iloc[2] == "extensive_displacement"
    assert pd.isna(merged["traj_cluster"].iloc[3])
