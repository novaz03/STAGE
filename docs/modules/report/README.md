# `evacmob.report`

Reporting utilities that mirror the neural models and diagnostics from the SES notebooks.

- `write_text_report` – ensure report directories exist and persist inline narratives.
- `build_bottleneck_mlp` / `train_or_load_model` – construct and train the bottleneck MLP used for POI embedding inference.
- `encode_features`, `collect_latents` – run inference and capture latent vectors for downstream analytics.
- `student_t_distribution`, `target_distribution` – helper maths for soft clustering targets.
- `plot_training_history` – generate the training-loss visual used in the notebooks.
