# `evacmob.simulate`

Simulation helpers derived from the `visualize_0822` notebook, making the cohort playbooks reusable outside Jupyter.

- `SimulationConfig` – configuration for cohort behaviour, sampling weights, and RNG seeds.
- `prepare_pois_with_counties` / `pick_local_metric_crs` – harmonise POIs with county polygons and identify an appropriate projection.
- `pick_poi`, `pick_from_any`, `ensure_min_errands` – category-aware movement logic matching the notebook playbooks.
- `simulate_person` – execute the daily decision rules for a single cohort member.
- `run_simulation` – batch simulation over a people dataframe, returning day/POI traces per person.
