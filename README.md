# hurricane-mobility-sim

Modularized repo scaffolded from a set of exploratory notebooks. This converts the uploaded notebooks into importable Python modules and sets up a clean package + CLI for simulation, visualization, and reporting.

## What you get

- `src/evacmob/`: Python package with core modules and auto-converted notebook code under `src/evacmob/notebooks/`.
- `notebooks/`: Original notebooks kept intact.
- `docs/reports/`: Holds HTML reports (e.g., `simulated_trajectories.html`).
- `scripts/`: Small CLIs and a Slurm template.
- `tests/`: Minimal tests & structure for pytest.
- CI via GitHub Actions and code quality via ruff/black (optional).

## Quickstart

```bash
# (Optional) conda env
conda env create -f environment.yml
conda activate evacmob

# install in editable mode
pip install -e .

# run the CLI
python scripts/evacmob_cli.py simulate --out outputs/sim.txt

# copy an uploaded HTML into docs
python scripts/evacmob_cli.py copy-html --src docs/reports/simulated_trajectories.html --dest docs/reports/simulated_trajectories.html
```

### Slurm example

```bash
sbatch scripts/slurm/run_pipeline.sbatch
```

### Recommended workflow

1. Move stable functions from `src/evacmob/notebooks/*.py` into `src/evacmob/` core modules.
2. Use `scripts/` as thin wrappers around package functions.
3. Add tests in `tests/` as you stabilize APIs.
4. Fill in project metadata in `pyproject.toml` and LICENSE as needed.

## Pushing to GitHub

```bash
git init
git add .
git commit -m "Initial commit: scaffolded from notebooks"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```
