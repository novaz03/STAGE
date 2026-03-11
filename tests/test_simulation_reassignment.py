import numpy as np
import pandas as pd

from evacmob.simulation_reassignment import (
    RAW_TO_SEMANTIC_LABEL,
    build_reassigned_label_table,
    merge_reassigned_labels,
    prepare_reassignment_features,
    reassign_simulation_labels,
)


class _DummyModel:
    feature_names_in_ = np.array(["f1", "f2", "f_missing"])

    def predict(self, X):
        # Return legacy labels so mapping can be verified.
        # Choose class by sign of f1.
        labels = []
        for val in X["f1"].fillna(0.0).to_numpy():
            if val > 0:
                labels.append("sip_home_grocery")
            elif val < 0:
                labels.append("sip_hospital")
            else:
                labels.append("evac_out_of_zone")
        return np.asarray(labels, dtype=object)


def test_prepare_reassignment_features_aligns_schema():
    df = pd.DataFrame({"traj_id": ["a", "b"], "f2": [True, False], "f1": [1.0, -2.0]})
    X = prepare_reassignment_features(df, _DummyModel())

    assert list(X.columns) == ["f1", "f2", "f_missing"]
    assert X["f2"].tolist() == [1, 0]
    assert X["f_missing"].isna().all()


def test_reassign_simulation_labels_writes_semantic_traj_cluster():
    df = pd.DataFrame({"traj_id": ["b", "a", "c"], "f1": [1.0, -1.0, 0.0], "f2": [1, 0, 1]})
    out = reassign_simulation_labels(df, model=_DummyModel())

    assert out["traj_id"].tolist() == ["a", "b", "c"]
    assert out["knn_k3_label_raw"].tolist() == [
        "sip_hospital",
        "sip_home_grocery",
        "evac_out_of_zone",
    ]
    assert out["traj_cluster"].tolist() == [
        RAW_TO_SEMANTIC_LABEL["sip_hospital"],
        RAW_TO_SEMANTIC_LABEL["sip_home_grocery"],
        RAW_TO_SEMANTIC_LABEL["evac_out_of_zone"],
    ]


def test_build_and_merge_reassigned_labels():
    reassigned = pd.DataFrame(
        {
            "traj_id": ["t1", "t2", "t3"],
            "traj_cluster": ["compact_local", "intermediate_directed", "extensive_displacement"],
        }
    )
    label_table = build_reassigned_label_table(reassigned)
    assert label_table.shape == (3, 2)

    points = pd.DataFrame({"traj_id": ["t1", "t1", "t3", "x"], "value": [1, 2, 3, 4]})
    merged = merge_reassigned_labels(points, label_table)
    assert merged["traj_cluster"].tolist() == [
        "compact_local",
        "compact_local",
        "extensive_displacement",
        np.nan,
    ]

