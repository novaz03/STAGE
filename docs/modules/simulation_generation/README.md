# `evacmob.simulation_generation`

Random trajectory generator extracted from `visualize_0822 (5).ipynb`.

- `generate_random_trajectories` – generates 300 people over 7 days with exactly 3 tuneable knobs:
  - `near_scale_m`
  - `far_scale_m`
  - `evac_min_dist_m`
- `save_trajectories_csv` – save step-level output to CSV.

Output labels are semantic CSV labels (not legacy notebook variable names):

- `compact_local`
- `intermediate_directed`
- `extensive_displacement`
