# STAGE: A Spatio-Temporal Attention and Graph Embedding Framework for Modeling Human Mobility Trajectories

STAGE reorganizes a collection of exploratory notebooks into a coherent Python package with reusable modules, documentation, and scripts for simulation, visualization, and reporting.

## File structure

- `src/evacmob/`: Core Python package plus notebook-derived reference code in `src/evacmob/notebooks/`.
- `docs/modules/`: Module-level READMEs describing responsibilities and APIs.
- `notebooks/`: Unmodified exploratory notebooks (optional reference).
- `docs/reports/`: Generated artefacts such as HTML summaries.
- `scripts/`: Command-line helpers and batch templates.
- `tests/`: Minimal pytest scaffolding.

### Module overview

Each module now has a dedicated README under `docs/modules/`:

- `data` – data ingest and spatial joins (`docs/modules/data/README.md`)
- `preprocess` – trip curation and feature engineering (`docs/modules/preprocess/README.md`)
- `simulate` – cohort playbooks and trajectory simulation (`docs/modules/simulate/README.md`)
- `report` – bottleneck MLP utilities and analytics (`docs/modules/report/README.md`)
- `visualize` – plotting helpers (`docs/modules/visualize/README.md`)
- `pipeline` – end-to-end orchestration (`docs/modules/pipeline/README.md`)

## Quickstart

```bash
# (Optional) conda env
conda env create -f environment.yml
conda activate evacmob

# install in editable mode
pip install -e .

# run the CLI
python scripts/evacmob_cli.py simulate --out outputs/sim.txt

# or run the streamlined pipeline (example)
python - <<'PY'
from evacmob.pipeline import PipelineConfig, run_pipeline

config = PipelineConfig(
    hex_path="Hex_tesse_raw.parquet",
    poi_path="POI_encoded_embeddings.parquet",
    poi_vector_col="z",
    visualization_path="outputs/hex_map.png",
    visualization_inset_bounds=(-82.5, 27.9, -82.3, 28.1),
    states_path="cb_2018_us_state_500k.shp",
    trip_logs_path="DRIVES_data.csv",            # optional raw trip logs (auto-detected format)
    trajectory_df_path="imputed_ses.parquet",    # or provide precomputed features
    trajectory_id_col="traj_id",
    # recompute_embeddings=True,
    # llm_model_name="sentence-transformers/all-MiniLM-L6-v2",
    # poi_embedding_text_col="llm_label",
)
run_pipeline(config)
PY
```

The simulation command writes `outputs/sim.txt` with `person_id,day,latitude,longitude`
rows generated from a built-in demo dataset so you can inspect the movement traces
without sourcing external files.

Enable `recompute_embeddings` with a Hugging Face model name to regenerate POI embeddings on the fly before the autoencoder stage (see `docs/modules/pipeline/README.md` for details).

## New pipeline walkthrough (data → POI latents)

The `new_pipeline/` package distils the notebooks into composable steps that can be
run from a clean checkout. The commands below assume you have the raw files the
notebooks relied on:

| Description | Example filename |
|-------------|------------------|
| Full raw POI export | `US_POI.csv` |
| POI subset already filtered to the study hull | `Hex_bound_POI.csv` |
| Hex tessellation for the study region | `Hex_tesse_raw.parquet` |
| Hurricane trajectory matrices / metadata (for concave hull + AE) | `hurricane_matrix.csv`, `simulated_traj_points.parquet` or real equivalents |

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

2. **Encode POIs with the fine-tuned LLM**

   Convert the raw POI table into the projection matrix of 1,152‑dim vectors:

   ```bash
   python -m new_pipeline.encode_poi_embeddings \
       --input-csv Hex_bound_POI.csv \
       --output-parquet POI_vec_proj_matrix.parquet \
       --checkpoint-dir /path/to/your/llm/checkpoint \
       --base-model google/gemma-3-1b-it
   ```

   *Artifacts*: `POI_vec_proj_matrix.parquet` (one row per POI with the `concatenated_vec` column).

3. **Aggregate POIs to hexagons / block groups**

   The updated `data_conversion (1).py` script uses the new aggregation helpers. It
   produces a GeoDataFrame that stores the averaged embedding per hex (`embedding`)
   and the POI count (`poi_count`):

   ```bash
   python "data_conversion (1).py"
   ```

   Adjust the `CONFIG` entries at the top of the script if your file layout differs.
   *Artifacts*: `POI_encoded_embeddings.parquet` with one row per hexagon (or CBG).

4. **Train the bottleneck MLP on POI vectors**

   This learns the supervised latent `z_poi` features that the notebooks used downstream:

   ```bash
   python - <<'PY'
   from new_pipeline.mlp import run_mlp_training, MLPConfig

   run_mlp_training(
       MLPConfig(
           input_parquet="POI_vec_proj_matrix.parquet",
           output_parquet="POI_vec_proj_matrix.parquet",  # overwrites with z_poi
           checkpoint_path="models/bottleneck_mlp.pth",
       )
   )
   PY
   ```

5. **Train the trajectory autoencoder (real or synthetic data)**

   Use the new CLI to run the Transformer autoencoder over either the real joined
   dataset or the simulated trajectories:

   ```bash
   python scripts/train_autoencoder.py \
       --mode real \
       --real-dataset GEOID_SES_point.parquet \
       --checkpoint models/trajectory_autoencoder.pth \
       --latents models/trajectory_latents.npz
   ```

   Replace `--mode real` with `--mode synthetic --synthetic-dataset simulated_traj_points.parquet`
   for the synthetic option. The command writes both the checkpoint and the latent matrix.

At this point you have:

- `POI_vec_proj_matrix.parquet` containing both `concatenated_vec` and the bottleneck latent
  `z_poi`.
- `POI_encoded_embeddings.parquet` with a single embedding per spatial cell.
- `models/bottleneck_mlp.pth` and `models/trajectory_autoencoder.pth` with the learned weights.

These artifacts feed back into the visualization or downstream analysis notebooks exactly as before—only the
data preparation is now scripted and reproducible.

### Slurm example

```bash
sbatch scripts/slurm/run_pipeline.sbatch
```
