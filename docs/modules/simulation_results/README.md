# `evacmob.simulation_results`

Utilities for converting simulation clustering output into a reproducible diagnostics view.

- `align_clusters_to_truth` – Hungarian matching to align cluster IDs to semantic labels.
- `evaluate_simulation_results` – computes contingency table, chi-square, Cramer's V, aligned confusion matrix, and classification metrics.
- `format_simulation_results` / `print_simulation_results` – notebook-style text summary rendering.
- `view_simulation_results` – one-shot evaluate + print helper.
- `view_simulation_results_from_stability` – convenience wrapper using `stability_result["consensus_labels"]`.
