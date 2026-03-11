import numpy as np

from evacmob.simulation_results import evaluate_simulation_results, format_simulation_results


def _build_labels():
    y_true = []
    clusters = []

    table = {
        "compact_local": {0: 50, 1: 40, 2: 10},
        "extensive_displacement": {0: 2, 1: 8, 2: 90},
        "intermediate_directed": {0: 15, 1: 75, 2: 10},
    }
    for cls_name, row in table.items():
        for cluster_id, count in row.items():
            y_true.extend([cls_name] * count)
            clusters.extend([cluster_id] * count)

    return np.asarray(y_true), np.asarray(clusters)


def test_evaluate_simulation_results_matches_expected_metrics():
    y_true, final_labels = _build_labels()
    X = np.zeros((len(y_true), 2), dtype=float)

    view = evaluate_simulation_results(X=X, y_true=y_true, final_labels=final_labels, best_k=3)

    assert view.best_k == 3
    assert view.cluster_sizes.to_dict() == {0: 67, 1: 123, 2: 110}
    assert view.cluster_to_truth_map == {
        2: "extensive_displacement",
        0: "compact_local",
        1: "intermediate_directed",
    }
    assert np.isclose(view.chi2, 226.3382, atol=1e-4)
    assert view.dof == 4
    assert np.isclose(view.p_value, 8.10668459819785e-48)
    assert np.isclose(view.cramers_v, 0.6141588181280092)
    assert np.isclose(view.accuracy, 0.7166666666666667)
    assert np.isclose(view.balanced_accuracy, 0.7166666666666667)

    report_text = format_simulation_results(view)
    assert "=== Raw contingency table: ground truth vs cluster ===" in report_text
    assert "=== Classification report ===" in report_text

