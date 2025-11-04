# Pipeline (new workflow)

The exploratory notebooks have been distilled into importable modules under `evacmob.pipeline`.
Each component can be executed independently, allowing you to re-run specific stages without
touching the rest of the workflow. Visualisation helpers continue to live in `evacmob.visualize`;
everything else is driven by the modules documented here.

## Modules

- `evacmob.pipeline.data` – read and clean raw POI exports, tokenise Hugging Face datasets, and
  chunk text for language-modelling tasks.
- `evacmob.pipeline.tokenizer` / `evacmob.pipeline.pipeline` – load base tokenisers, extend them with
  domain special tokens, and fine-tune a causal LM (Gemma + LoRA by default).
- `evacmob.pipeline.encode_poi_embeddings` – CLI/script entry point that turns the cleaned POI CSV
  into `POI_vec_proj_matrix.parquet` using the fine-tuned checkpoint.
- `evacmob.pipeline.aggregation` – utilities to mean-pool POI vectors into hexagon / block-group
  embeddings.
- `evacmob.pipeline.mlp` – trains the bottleneck classifier that produced the `z_poi` latents in the
  notebooks. The helper writes the enriched parquet and saves the checkpoint.
- `evacmob.pipeline.autoencoder` – builds the Transformer autoencoder used for trajectory latent
  modelling (supports real and synthetic inputs).

Visualisation helpers still rely on the existing `evacmob.visualize` module; everything else
should prefer the `evacmob.pipeline` implementations.

## Typical flow

1. `python -m evacmob.pipeline.pipeline` – fine-tune the causal LM with LoRA on the POI text.
2. `evacmob.pipeline.encode_poi_embeddings.encode_poi_to_parquet` – from raw POI CSV to
   `POI_vec_proj_matrix.parquet`.
3. `python -m evacmob.pipeline.mlp` – fit the bottleneck MLP; adds the supervised `z_poi`
   latent to the projection parquet and saves a checkpoint.
4. `python -m evacmob.pipeline.aggregation` – assign POIs to hexes / CBGs and
   compute averaged embeddings per cell (expects the `z_poi` column from the previous step).
5. `evacmob.pipeline.train_autoencoder` – train the Transformer autoencoder, writing both
   the model checkpoint and the latent matrix.

See the top-level `README.md` for concrete command examples.
