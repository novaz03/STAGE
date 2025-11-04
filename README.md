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

## Environment variables

Copy `.env.example` to `.env` (ignored by git) and populate the placeholders before running any workflows that interact with the Hugging Face Hub:

```bash
cp .env.example .env
echo "HF_TOKEN=your_real_token" >> .env   # or edit with your editor of choice
```

All scripts that require the token will read it from the `HF_TOKEN` environment variable.

### Slurm example

```bash
sbatch scripts/slurm/run_pipeline.sbatch
```
