# STAGE: A Spatio-Temporal Attention and Graph Embedding Framework for Modeling Human Mobility Trajectories

STAGE reorganizes a collection of exploratory notebooks into a coherent Python package with reusable modules, documentation, and scripts for analysis, visualization, and reporting.

## File structure

- `src/evacmob/`: Core Python package plus notebook-derived reference code in `src/evacmob/notebooks/`.
- `docs/modules/`: Module-level READMEs describing responsibilities and APIs.
- `notebooks/`: Unmodified exploratory notebooks (optional reference).
- `scripts/`: Command-line helpers and batch templates.
- `tests/`: Minimal pytest scaffolding.

### Module overview

Each module now has a dedicated README under `docs/modules/`:

- `data` – data ingest and spatial joins (`docs/modules/data/README.md`)
- `preprocess` – trip curation and feature engineering (`docs/modules/preprocess/README.md`)
- `report` – bottleneck MLP utilities and analytics (`docs/modules/report/README.md`)
- `simulation_generation` – random trajectory generation (`docs/modules/simulation_generation/README.md`)
- `simulation_label_assignment` – model-based simulation label assignment via `knn_k3.joblib` (`docs/modules/simulation_label_assignment/README.md`)
- `simulation_results` – clustering-vs-truth diagnostics for simulation outputs (`docs/modules/simulation_results/README.md`)
- `visualize` – plotting helpers (`docs/modules/visualize/README.md`)
- `pipeline` – end-to-end orchestration (`docs/modules/pipeline/README.md`)

## Quickstart

```bash
conda env create -f environment.yaml
conda activate new-pipeline

pip install -e .
```

## New pipeline walkthrough (data → POI latents)

The pipeline code now lives directly under `evacmob.pipeline` and mirrors the notebook
experiments in a composable fashion. The commands below assume you have the raw files the
notebooks relied on:

| Description | Example filename |
|-------------|------------------|
| Full raw POI export | `US_POI.csv` |
| POI subset already filtered to the study hull | `Hex_bound_POI.csv` |
| Hex tessellation for the study region | `Hex_tesse_raw.parquet` |
| Hurricane trajectory matrices / metadata (for concave hull + AE) | `hurricane_matrix.csv` |

1. **Create the environment**

   ```bash
   conda env create -f environment.yaml
   conda activate new-pipeline
   ```

   Copy `.env.example` to `.env`, populate `HF_TOKEN`, and export it before running any
   Hugging Face operations:

   ```bash
   cp .env.example .env
   export $(grep -v '^#' .env | xargs)  # or source the file in your shell
   ```

2. **Generate study-area inputs (optional)**

   ```bash
   python scripts/generate_hex_inputs.py \
       --raw-poi US_POI.csv \
       --filtered-poi Hex_bound_POI.csv \
       --hex-parquet Hex_tesse_raw.parquet \
       --min-lon -88.57 --max-lon -79.95 \
       --min-lat 24.45 --max-lat 32.35 \
       --hex-radius-m 8000
   ```

   This filters the raw POI export and builds an initial hex tessellation if you
   don’t already have `Hex_bound_POI.csv` / `Hex_tesse_raw.parquet`.  If your
   `US_POI.csv` already contains a WKT geometry column, pass
   `--geometry-column GEOMETRY` (or whichever column name you use) and the tool
   will derive the concave hull directly from that geometry instead of applying
   the bounding-box filter.  The default configuration assumes the SafeGraph
   header (e.g. `PLACEKEY,...,LATITUDE,LONGITUDE,...,GEOMETRY_TYPE`), so
   latitude/longitude columns must be present when a geometry column is not.

   If you prefer census block group tessellation, use the CBG loader:

   ```bash
   python - <<'PY'
   from evacmob.pipeline import CBGLoadConfig, load_cbgses

   cfg = CBGLoadConfig(
       attributes_path="bg_fl_2022.xlsx",
       geometry_path="fl_bg.geojson",
       geometry_crs="EPSG:4326",
   )
   ses_gdf = load_cbgses(cfg)
   ses_gdf.to_parquet("CBG_tessellation.parquet")
   PY
   ```

   The downstream aggregation step accepts either the hexagon parquet or the CBG parquet.

3. **Fine-tune the causal language model (LoRA)**

   If you need to adapt the base Gemma checkpoint to your POI corpus, run the LoRA
   fine-tuning step first (skip this if you already have a tuned checkpoint):

   ```bash
   python -m evacmob.pipeline.pipeline \
       --data-csv US_POI.csv \
       --output-dir models/llm_finetune_run \
       --tokenizer-base google/gemma-3-1b-it \
       --model-name google/gemma-3-1b-it \
       --epochs 1 \
       --per-device-batch-size 256 \
       --gradient-steps 64 \
       --block-size 64
   ```

   *Artifacts*: updated tokenizer and LoRA weights under `models/llm_finetune_run/`.

4. **Encode POIs with the fine-tuned LLM**

   Convert the raw POI table into the projection matrix of 1,152‑dim vectors.
   Point `--checkpoint-dir` to the folder produced in Step 2 (or your own checkpoint):

   ```bash
   python -m evacmob.pipeline.encode_poi_embeddings \
       --input-csv Hex_bound_POI.csv \
       --output-parquet POI_vec_proj_matrix.parquet \
       --checkpoint-dir models/llm_finetune_run \
       --base-model google/gemma-3-1b-it
   ```

   *Artifacts*: `POI_vec_proj_matrix.parquet` (one row per POI with the `concatenated_vec` column).

5. **Train the bottleneck MLP on POI vectors**

   This learns the supervised latent `z_poi` features that the notebooks used downstream:

   ```bash
   python -m evacmob.pipeline.mlp \
       --input-parquet POI_vec_proj_matrix.parquet \
       --checkpoint-path models/bottleneck_mlp.pth
   ```

6. **Aggregate POIs to hexagons / block groups**

   This step expects the `z_poi` column written by Step 3.

   ```bash
   python -m evacmob.pipeline.aggregation \
       --poi-geometry-csv Hex_bound_POI.csv \
       --poi-parquet POI_vec_proj_matrix.parquet \
       --hex-parquet Hex_tesse_raw.parquet \
       --output POI_encoded_embeddings.parquet \
       --latent-column z_poi
   ```

   *Artifacts*: `POI_encoded_embeddings.parquet` with one row per hexagon (or CBG).

7. **Train the trajectory autoencoder**

   Use the CLI helper to run the Transformer autoencoder over the joined trajectory dataset:

   ```bash
   python scripts/train_autoencoder.py \
       --dataset GEOID_SES_point.parquet \
       --checkpoint models/trajectory_autoencoder.pth \
       --latents models/trajectory_latents.npz
   ```

   The command writes both the checkpoint and the latent matrix.

At this point you have:

- `POI_vec_proj_matrix.parquet` containing both `concatenated_vec` and the bottleneck latent `z_poi`.
- `POI_encoded_embeddings.parquet` with a single embedding per spatial cell.
- `models/bottleneck_mlp.pth` and `models/trajectory_autoencoder.pth` with the learned weights.

These artifacts feed back into the visualization or downstream analysis notebooks exactly as before—only the
data preparation is now scripted and reproducible.

## Simulation input downsampling

If you want a smaller labeled simulation slice, use the helper below. By default it
keeps `reference_lab` counts at `compact_local=25`, `intermediate_directed=25`,
and `extensive_displacement=75`, and applies the same `traj_id` subset to the
point-level parquet when present.

```bash
python scripts/subset_simulation_inputs.py
```

Override the counts explicitly if needed:

```bash
python scripts/subset_simulation_inputs.py \
    --compact-local 25 \
    --intermediate-directed 25 \
    --extensive-displacement 75
```

## Simulation clustering

To estimate the best clustering correctness on a labeled subset, run the repeated
KMeans search helper below. It tries multiple `k` values and random seeds, then
reports the best aligned accuracy against `reference_lab`.

```bash
python scripts/cluster_simulation_subset.py \
    --input hourly_locations_wide_25_25_75.csv \
    --k-values 2,3,4,5,6 \
    --n-seeds 20 \
    --score accuracy
```

You can also point it at the point-level parquet. In that case it first aggregates
the point rows into one trajectory-level feature vector per `traj_id`.

```bash
python scripts/cluster_simulation_subset.py \
    --input ref_simulation_point_gdf_25_25_75.parquet \
    --k-values 2,3,4,5,6 \
    --n-seeds 20
```

### Slurm example

```bash
sbatch scripts/slurm/run_pipeline.sbatch
```
