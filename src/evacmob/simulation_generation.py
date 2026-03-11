"""Random trajectory generation for the Hurricane simulation workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

# Public semantic labels used in CSV exports.
LABEL_COMPACT_LOCAL = "compact_local"
LABEL_INTERMEDIATE_DIRECTED = "intermediate_directed"
LABEL_EXTENSIVE_DISPLACEMENT = "extensive_displacement"

# Legacy notebook cohort names for reference only.
LEGACY_TO_SEMANTIC_LABEL = {
    "sip_home_grocery": LABEL_COMPACT_LOCAL,
    "sip_hospital": LABEL_INTERMEDIATE_DIRECTED,
    "evac_out_of_zone": LABEL_EXTENSIVE_DISPLACEMENT,
}

# Fixed simulation settings. Only distance scales are exposed as tweakable parameters.
_N_PEOPLE = 300
_SIMULATION_DAYS = 7
_LANDFALL_DAYS = {2, 3}
_MAX_CANDIDATES_FAST = 500
_DRIFT_PROB = 0.30
_DRIFT_SCALE_M = 5_000.0
_RANDOM_SEED = 42

# POI categories expected from the notebook dataset.
_CAT_HOME = "Activities Related to Real Estate"
_CAT_GROC = "Grocery Stores"
_CAT_GAS = "Gasoline Stations"
_CAT_BUILD = "Building Material and Supplies Dealers"
_CAT_HOSP = "Hospitals"
_CAT_HOTL = "Traveler Accommodation"
_CAT_HEALTH_STORE = "Health and Personal Care Stores"
_CAT_RESTAURANT = "Restaurants and Other Eating Places"


@dataclass(frozen=True)
class GeneratedTrajectories:
    """Simulation output container."""

    people_df: pd.DataFrame
    trajectories_df: pd.DataFrame
    full_trajectories: dict[int, list[tuple[int, int]]]
    pois_all: gpd.GeoDataFrame


def _pick_local_metric_crs(gdf_ll_4326: gpd.GeoDataFrame) -> str:
    centroid = gdf_ll_4326.to_crs(4326).unary_union.centroid
    lon = float(centroid.x)
    lat = float(centroid.y)
    zone = int((lon + 180) // 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return f"EPSG:{epsg}"


def _compact_consecutive(seq: list[int]) -> list[int]:
    if not seq:
        return []
    return [seq[0]] + [val for i, val in enumerate(seq[1:]) if val != seq[i]]


def generate_random_trajectories(
    gdf_proj: gpd.GeoDataFrame,
    ian_zone_gdf: gpd.GeoDataFrame,
    *,
    near_scale_m: float = 3_000.0,
    far_scale_m: float = 75_000.0,
    evac_min_dist_m: float = 50_000.0,
) -> GeneratedTrajectories:
    """Generate synthetic trajectories with only 3 distance knobs.

    The 3 semantic cohorts are directly labeled as:
    - ``compact_local``
    - ``intermediate_directed``
    - ``extensive_displacement``
    """
    if "cat_name" not in gdf_proj.columns:
        raise KeyError("gdf_proj must contain a 'cat_name' column.")
    if "County" not in ian_zone_gdf.columns:
        raise KeyError("ian_zone_gdf must contain a 'County' column.")

    rng = np.random.default_rng(_RANDOM_SEED)

    pois = gdf_proj.copy()
    if pois.crs is None:
        pois = pois.set_crs(4326)
    metric_crs = pois.crs
    if metric_crs.to_epsg() == 4326:
        metric_crs = _pick_local_metric_crs(pois)
        pois = pois.to_crs(metric_crs)

    zone = ian_zone_gdf.copy()
    if zone.crs is None:
        zone = zone.set_crs(4326)
    if zone.crs != metric_crs:
        zone = zone.to_crs(metric_crs)

    joined = gpd.sjoin(
        pois,
        zone[["County", "geometry"]].rename(columns={"County": "_zone_county"}),
        how="left",
        predicate="intersects",
    ).drop(columns=["index_right"], errors="ignore")
    pois_all = joined.reset_index(drop=True)

    in_zone_mask = pois_all["_zone_county"].notna()
    out_zone_mask = ~in_zone_mask

    cat_values = pois_all["cat_name"].fillna("").astype(str).to_numpy()
    all_categories = set(cat_values) - {""}

    def present(cats: list[str]) -> list[str]:
        return [cat for cat in cats if cat in all_categories]

    pool_compact_early = present([_CAT_GROC])
    pool_compact_late = present([_CAT_BUILD])
    pool_intermediate_main = present([_CAT_HOSP])
    pool_intermediate_support = present([_CAT_HEALTH_STORE])
    pool_extensive_prep = present([_CAT_GAS])
    pool_extensive_dest = present([_CAT_HOTL, _CAT_RESTAURANT])

    poi_positions = np.arange(len(pois_all))
    idx_in = poi_positions[in_zone_mask.to_numpy()]
    idx_out = poi_positions[out_zone_mask.to_numpy()]

    cat_to_idx_in = {cat: idx_in[cat_values[idx_in] == cat] for cat in all_categories}
    cat_to_idx_out = {cat: idx_out[cat_values[idx_out] == cat] for cat in all_categories}
    candidate_cache: dict[tuple[str, tuple[str, ...]], np.ndarray] = {}

    poi_xy = np.column_stack((pois_all.geometry.x.to_numpy(), pois_all.geometry.y.to_numpy()))

    def get_candidates(zone_key: str, cats: list[str] | None) -> np.ndarray:
        if cats is None:
            return idx_in if zone_key == "in" else idx_out
        key = (zone_key, tuple(sorted(cats)))
        if key in candidate_cache:
            return candidate_cache[key]
        lookup = cat_to_idx_in if zone_key == "in" else cat_to_idx_out
        arrays = [lookup.get(cat, np.array([], dtype=int)) for cat in cats]
        merged = np.unique(np.concatenate(arrays)) if arrays else np.array([], dtype=int)
        candidate_cache[key] = merged
        return merged

    def pick_strict(
        ref_idx: int,
        cats: list[str] | None,
        zone_key: str,
        mode: str,
        scale_m: float,
        min_dist_m: float = 0.0,
    ) -> int:
        candidates = get_candidates(zone_key, cats)
        if candidates.size == 0:
            return int(ref_idx)
        if candidates.size > _MAX_CANDIDATES_FAST:
            candidates = rng.choice(candidates, size=_MAX_CANDIDATES_FAST, replace=False)

        ref_xy = poi_xy[int(ref_idx)]
        cand_xy = poi_xy[candidates]
        dist = np.sqrt(((cand_xy - ref_xy) ** 2).sum(axis=1))

        if mode == "near":
            logits = -(dist / max(scale_m, 1.0))
            logits = logits - logits.max()
            weights = np.exp(logits)
        elif mode == "far":
            mask = dist >= min_dist_m
            if not mask.any():
                return int(ref_idx)
            logits = (dist[mask] / max(scale_m, 1.0))
            logits = logits - logits.max()
            weights = np.zeros_like(dist)
            weights[mask] = np.exp(logits)
        else:
            raise ValueError("mode must be 'near' or 'far'.")

        total = weights.sum()
        if total <= 0:
            return int(ref_idx)
        weights = weights / total
        return int(rng.choice(candidates, p=weights))

    home_positions = idx_in[cat_values[idx_in] == _CAT_HOME]
    if home_positions.size == 0:
        raise ValueError(
            "No in-zone home POIs found for category 'Activities Related to Real Estate'."
        )

    initial_positions = rng.choice(home_positions, size=_N_PEOPLE, replace=True)
    people_df = pd.DataFrame(
        {
            "person_id": np.arange(_N_PEOPLE, dtype=int),
            "current_location_idx": initial_positions.astype(int),
        }
    )
    people_df["home_location_idx"] = people_df["current_location_idx"]
    people_df["home_county"] = people_df["home_location_idx"].map(pois_all["_zone_county"])

    base, rem = divmod(_N_PEOPLE, 3)
    labels = (
        [LABEL_COMPACT_LOCAL] * (base + (1 if rem > 0 else 0))
        + [LABEL_INTERMEDIATE_DIRECTED] * (base + (1 if rem > 1 else 0))
        + [LABEL_EXTENSIVE_DISPLACEMENT] * base
    )
    labels_arr = np.asarray(labels, dtype=object)
    rng.shuffle(labels_arr)
    people_df["traj_cluster"] = labels_arr

    def simulate_one_person(home_idx: int, cluster: str) -> list[tuple[int, int]]:
        path: list[tuple[int, int]] = []
        current_loc = int(home_idx)

        for day in range(1, _SIMULATION_DAYS + 1):
            steps_today: list[int] = [current_loc]

            if cluster == LABEL_COMPACT_LOCAL:
                pool = pool_compact_early if day <= 2 else pool_compact_late
                if pool:
                    dest = pick_strict(
                        ref_idx=int(home_idx),
                        cats=pool,
                        zone_key="in",
                        mode="near",
                        scale_m=near_scale_m,
                    )
                    steps_today.extend([dest, int(home_idx)])
                current_loc = int(home_idx)

            elif cluster == LABEL_INTERMEDIATE_DIRECTED:
                if day == 1:
                    if pool_intermediate_support:
                        prep = pick_strict(
                            ref_idx=int(home_idx),
                            cats=pool_intermediate_support,
                            zone_key="in",
                            mode="near",
                            scale_m=near_scale_m,
                        )
                        steps_today.extend([prep, int(home_idx)])
                elif day in _LANDFALL_DAYS:
                    hospital = pick_strict(
                        ref_idx=int(home_idx),
                        cats=pool_intermediate_main,
                        zone_key="in",
                        mode="near",
                        scale_m=near_scale_m,
                    )
                    steps_today.append(hospital)
                    current_loc = hospital
                elif day == max(_LANDFALL_DAYS) + 2:
                    steps_today.append(int(home_idx))
                    current_loc = int(home_idx)
                else:
                    steps_today.append(current_loc)

            elif cluster == LABEL_EXTENSIVE_DISPLACEMENT:
                if day == 1:
                    gas = pick_strict(
                        ref_idx=int(home_idx),
                        cats=pool_extensive_prep,
                        zone_key="in",
                        mode="near",
                        scale_m=near_scale_m,
                    )
                    steps_today.extend([gas, int(home_idx)])
                    current_loc = int(home_idx)
                elif day == 2:
                    evac_dest = pick_strict(
                        ref_idx=int(home_idx),
                        cats=pool_extensive_dest,
                        zone_key="out",
                        mode="far",
                        scale_m=far_scale_m,
                        min_dist_m=evac_min_dist_m,
                    )
                    steps_today.append(evac_dest)
                    current_loc = evac_dest
                else:
                    if rng.random() < _DRIFT_PROB:
                        local_dest = pick_strict(
                            ref_idx=current_loc,
                            cats=pool_extensive_dest,
                            zone_key="out",
                            mode="near",
                            scale_m=_DRIFT_SCALE_M,
                        )
                        steps_today.extend([local_dest, current_loc])
                    else:
                        steps_today.append(current_loc)

            for poi_idx in _compact_consecutive(steps_today):
                path.append((day, int(poi_idx)))

        return path

    full_trajectories: dict[int, list[tuple[int, int]]] = {}
    for row in people_df.itertuples(index=False):
        full_trajectories[int(row.person_id)] = simulate_one_person(
            home_idx=int(row.current_location_idx),
            cluster=str(row.traj_cluster),
        )

    pid_to_cluster = people_df.set_index("person_id")["traj_cluster"]
    pid_to_home = people_df.set_index("person_id")["home_location_idx"]
    rows: list[dict[str, object]] = []
    for pid, path in full_trajectories.items():
        home_idx = int(pid_to_home.loc[pid])
        for order, (day, poi_idx) in enumerate(path):
            poi = pois_all.iloc[int(poi_idx)]
            rows.append(
                {
                    "person_id": int(pid),
                    "traj_cluster": str(pid_to_cluster.loc[pid]),
                    "day": int(day),
                    "step_order": int(order),
                    "poi_index": int(poi_idx),
                    "poi_category": poi.get("cat_name"),
                    "county": poi.get("_zone_county"),
                    "is_home": bool(poi_idx == home_idx),
                }
            )

    trajectories_df = pd.DataFrame(rows)
    return GeneratedTrajectories(
        people_df=people_df,
        trajectories_df=trajectories_df,
        full_trajectories=full_trajectories,
        pois_all=pois_all,
    )


def save_trajectories_csv(df: pd.DataFrame, out_path: str | Path) -> Path:
    """Save generated trajectory rows to CSV."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
