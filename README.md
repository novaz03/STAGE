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
conda env create -f environment.yaml
conda activate new-pipeline

pip install -e .

# run the CLI
python scripts/evacmob_cli.py simulate --out outputs/sim.txt
```

The simulation command writes `outputs/sim.txt` with `person_id,day,latitude,longitude`
rows generated from a built-in demo dataset so you can inspect the movement traces
without sourcing external files.

## New pipeline walkthrough (data → POI latents)

The pipeline code now lives directly under `evacmob.pipeline` and mirrors the notebook
experiments in a composable fashion. The commands below assume you have the raw files the
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

2. **Fine-tune the causal language model (LoRA)**

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

3. **Encode POIs with the fine-tuned LLM**

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

4. **Train the bottleneck MLP on POI vectors**

   This learns the supervised latent `z_poi` features that the notebooks used downstream:

   ```bash
   python -m evacmob.pipeline.mlp \
       --input-parquet POI_vec_proj_matrix.parquet \
       --checkpoint-path models/bottleneck_mlp.pth
   ```

5. **Aggregate POIs to hexagons / block groups**

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

6. **Train the trajectory autoencoder (real or synthetic data)**

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

- `POI_vec_proj_matrix.parquet` containing both `concatenated_vec` and the bottleneck latent `z_poi`.
- `POI_encoded_embeddings.parquet` with a single embedding per spatial cell.
- `models/bottleneck_mlp.pth` and `models/trajectory_autoencoder.pth` with the learned weights.

These artifacts feed back into the visualization or downstream analysis notebooks exactly as before—only the
data preparation is now scripted and reproducible.

### Slurm example

```bash
sbatch scripts/slurm/run_pipeline.sbatch
```
