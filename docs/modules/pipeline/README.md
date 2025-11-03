# `evacmob.pipeline`

The pipeline package is now split across focused submodules while keeping the public API available through `evacmob.pipeline`.

- `config` – `PipelineConfig`, `PipelineArtifacts` dataclasses.
- `core` – the `run_pipeline` orchestration entry point.
- `trips` – `load_trip_logs`, `preprocess_trip_logs`, `build_trajectory_features_from_segments`.
- `llm` – `evaluate_poi_labels_with_llm`, `load_pretrained_llm`, `embed_texts_with_llm`, `recompute_poi_embeddings_with_llm`.
- `features` – `train_autoencoder_embeddings`, `aggregate_latents_to_hex`, `postprocess_hex_features`, `project_trajectories`, `cluster_latents`.

### LLM-driven embeddings

Set `PipelineConfig.recompute_embeddings=True`, along with `llm_model_name` (and optional tokenizer/device overrides), to regenerate POI embeddings from notebook text fields at runtime. The pipeline mean-pools the model’s last hidden state with optional L2 normalisation before handing the vectors to the autoencoder stage.
