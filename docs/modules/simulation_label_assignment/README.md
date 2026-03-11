# `evacmob.simulation_label_assignment`

Model-based simulation label assignment utilities built around `knn_k3.joblib`.

- `assign_simulation_labels` – applies the saved model and writes assigned semantic labels
  into `traj_cluster`.
- `prepare_label_assignment_features` – aligns feature columns to model `feature_names_in_`.
- `build_assigned_label_table` – one row per `traj_id`.
- `merge_assigned_labels` – joins assigned labels into other trajectory/point tables.
- `save_assigned_labels` – writes CSV/Parquet outputs.

Assigned semantic labels:

- `compact_local`
- `intermediate_directed`
- `extensive_displacement`
