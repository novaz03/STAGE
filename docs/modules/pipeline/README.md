# Pipeline (new workflow)

The exploratory notebooks have been distilled into importable modules under `new_pipeline/`.
Each component can be executed independently, allowing you to re-run specific stages without
touching the rest of the workflow. The legacy `evacmob.pipeline` package still exists for the
visualisation wrappers, but all data-processing / modelling code now lives here.

## Modules

- `new_pipeline.data` – read and clean raw POI exports, tokenise Hugging Face datasets, and
  chunk text for language-modelling tasks.
- `new_pipeline.tokenizer` / `new_pipeline.pipeline` – load base tokenisers, extend them with
  domain special tokens, and fine-tune a causal LM (Gemma + LoRA by default).
- `new_pipeline.encode_poi_embeddings` – CLI/script entry point that turns the cleaned POI CSV
  into `POI_vec_proj_matrix.parquet` using the fine-tuned checkpoint.
- `new_pipeline.aggregation` – utilities to mean-pool POI vectors into hexagon / block-group
  embeddings.
- `new_pipeline.mlp` – trains the bottleneck classifier that produced the `z_poi` latents in the
  notebooks. The helper writes the enriched parquet and saves the checkpoint.
- `new_pipeline.autoencoder` – builds the Transformer autoencoder used for trajectory latent
  modelling (supports real and synthetic inputs).

Visualisation helpers still rely on the existing `evacmob.visualize` module; everything else
should prefer the `new_pipeline` implementations.

## Typical flow

1. `new_pipeline.encode_poi_embeddings.encode_poi_to_parquet` – from raw POI CSV to
   `POI_vec_proj_matrix.parquet`.
2. `data_conversion (1).py` (imports `new_pipeline.aggregation`) – assign POIs to hexes / CBGs and
   compute averaged embeddings per cell.
3. `new_pipeline.mlp.run_mlp_training` – fit the bottleneck MLP; adds the supervised `z_poi`
   latent to the projection parquet and saves a checkpoint.
4. `new_pipeline.autoencoder.train_autoencoder` – train the Transformer autoencoder, writing both
   the model checkpoint and the latent matrix.

See the top-level `README.md` for concrete command examples.
