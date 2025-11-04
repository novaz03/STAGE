"""Simulation helpers aligned with ``simulation/simulation.ipynb``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point


@dataclass(frozen=True)
class Archetype:
    """Encapsulate the ordered category pattern for a mobility archetype."""

    pattern: Sequence[str]
    start_categories: Sequence[str] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.pattern, (str, bytes)):
            raise TypeError("pattern must be a sequence of category names.")
        object.__setattr__(self, "pattern", tuple(self.pattern))

        if self.start_categories is None:
            return
        if isinstance(self.start_categories, (str, bytes)):
            object.__setattr__(self, "start_categories", (self.start_categories,))
        else:
            object.__setattr__(self, "start_categories", tuple(self.start_categories))


@dataclass
class SimulationConfig:
    """Configuration bundle mirroring the notebook constants and playbooks."""

    distance_scale: float = 20_000.0
    county_weight_multiplier: Dict[str, float] = field(
        default_factory=lambda: {"same_county": 1.0, "diff_county": 3.0}
    )
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(42))
    pre_landfall_action_weights: Sequence[float] = (0.7, 0.3)
    total_days: int = 7
    impacted_counties: Sequence[str] = field(
        default_factory=lambda: (
            "Lee",
            "Charlotte",
            "Sarasota",
            "Collier",
            "DeSoto",
            "Hardee",
            "Manatee",
            "Hillsborough",
            "Pinellas",
        )
    )
    home_category: str = "Activities Related to Real Estate"
    category_column: str = "cat_name"
    county_column: str = "County"
    weight_eps: float = 1e-9
    archetypes: Mapping[str, Archetype] = field(
        default_factory=lambda: {
            "prepping": Archetype(
                pattern=(
                    "Grocery Stores",
                    "Gasoline Stations",
                    "Building Material and Supplies Dealers",
                    "Activities Related to Real Estate",
                )
            ),
            "evacuating": Archetype(
                pattern=(
                    "Gasoline Stations",
                    "Traveler Accommodation",
                )
            ),
            "sheltering_in_place": Archetype(
                pattern=("Hospitals",),
                start_categories=("Activities Related to Real Estate", "Traveler Accommodation"),
            ),
            "returning_home": Archetype(
                pattern=("Gasoline Stations", "Activities Related to Real Estate"),
                start_categories=("Traveler Accommodation",),
            ),
            "post_storm_recovery": Archetype(
                pattern=(
                    "Grocery Stores",
                    "Building Material and Supplies Dealers",
                    "Activities Related to Real Estate",
                )
            ),
        }
    )
    _pre_landfall_probs: Sequence[float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        probs = np.asarray(self.pre_landfall_action_weights, dtype=float)
        if probs.ndim != 1 or probs.size != 2:
            raise ValueError("pre_landfall_action_weights must contain two entries.")
        if np.any(probs < 0):
            raise ValueError("pre_landfall_action_weights must be non-negative.")
        total = probs.sum()
        if total <= 0:
            raise ValueError("pre_landfall_action_weights must sum to a positive value.")
        self._pre_landfall_probs = tuple((probs / total).tolist())

        self.impacted_counties = tuple(self.impacted_counties)

        normalised: Dict[str, Archetype] = {}
        for name, value in self.archetypes.items():
            if isinstance(value, Archetype):
                normalised[name] = value
            elif isinstance(value, Mapping):
                if "pattern" not in value:
                    raise ValueError(f"Archetype '{name}' is missing a 'pattern' entry.")
                normalised[name] = Archetype(
                    pattern=value["pattern"],
                    start_categories=value.get("start_cat") or value.get("start_categories"),
                )
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                normalised[name] = Archetype(pattern=value)
            else:
                raise TypeError(f"Unsupported archetype specification for '{name}'.")
        self.archetypes = normalised

    def pre_landfall_probs(self) -> np.ndarray:
        """Return the normalised probabilities for pre-landfall behaviour."""
        return np.asarray(self._pre_landfall_probs, dtype=float)


def pick_local_metric_crs(gdf_ll_4326: gpd.GeoDataFrame) -> str:
    """Select a local UTM projection for the supplied geographic geometries."""
    gdf = gdf_ll_4326.to_crs(4326)
    center = gdf.unary_union.centroid
    lon, lat = float(center.x), float(center.y)
    zone = int((lon + 180) // 6) + 1
    return f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}"


def prepare_pois_with_counties(
    pois: gpd.GeoDataFrame,
    counties: gpd.GeoDataFrame,
    source_field: str = "County",
    dest_field: str = "County",
    predicate: str = "within",
) -> gpd.GeoDataFrame:
    """Attach county names to POIs, mirroring the notebook's spatial join."""
    if pois.crs is None:
        raise ValueError("POI GeoDataFrame must have a CRS.")

    if counties.crs is None:
        counties = counties.set_crs(pois.crs)
    elif counties.crs != pois.crs:
        counties = counties.to_crs(pois.crs)

    if source_field not in counties.columns:
        if dest_field in counties.columns:
            source_field = dest_field
        elif "NAME" in counties.columns:
            source_field = "NAME"
        else:
            raise ValueError(
                "County GeoDataFrame must include a county name column "
                f"('{source_field}', '{dest_field}', or 'NAME')."
            )

    joined = gpd.sjoin(
        pois,
        counties[[source_field, "geometry"]],
        how="left",
        predicate=predicate,
    )

    return joined.rename(columns={source_field: dest_field}).drop(columns=["index_right"], errors="ignore")


def filter_home_locations(pois: gpd.GeoDataFrame, config: SimulationConfig) -> gpd.GeoDataFrame:
    """Return candidate home POIs constrained to impacted counties."""
    if config.category_column not in pois.columns:
        raise ValueError(f"POIs missing '{config.category_column}' column.")

    category_mask = pois[config.category_column] == config.home_category

    if config.county_column in pois.columns and config.impacted_counties:
        county_mask = pois[config.county_column].isin(config.impacted_counties)
    else:
        county_mask = True

    return pois[category_mask & county_mask]


def sample_population(
    pois: gpd.GeoDataFrame,
    population_by_county: Mapping[str, float] | pd.Series,
    n_people: int,
    config: SimulationConfig | None = None,
) -> pd.DataFrame:
    """Replicate the notebook's population seeding logic for the impact zone."""
    if n_people <= 0:
        raise ValueError("n_people must be a positive integer.")

    config = config or SimulationConfig()
    rng = config.rng

    population = pd.Series(population_by_county, dtype=float)
    population = population.loc[population.index.isin(config.impacted_counties)].dropna()
    if population.empty:
        raise ValueError("No population entries match the impacted counties.")

    probabilities = (population / population.sum()).to_numpy()
    counties = population.index.to_numpy()

    candidates = filter_home_locations(pois, config)
    if candidates.empty:
        raise ValueError("No candidate home POIs found within the impacted counties.")

    candidates_by_county: Dict[str, np.ndarray] = {}
    if config.county_column in candidates.columns:
        for county in counties:
            mask = candidates[config.county_column] == county
            candidates_by_county[county] = candidates.index[mask].to_numpy()

    fallback_pool = candidates.index.to_numpy()

    chosen_indices: List[int] = []
    for _ in range(n_people):
        county = str(rng.choice(counties, p=probabilities))
        pool = candidates_by_county.get(county)
        if pool is None or len(pool) == 0:
            pool = fallback_pool
        chosen_indices.append(int(rng.choice(pool)))

    people_df = pd.DataFrame(
        {
            "person_id": np.arange(n_people, dtype=int),
            "status": "at_home",
            "current_location_idx": chosen_indices,
            "home_location_idx": chosen_indices,
        }
    )

    if config.county_column in pois.columns:
        people_df["home_county"] = [pois.at[idx, config.county_column] for idx in chosen_indices]

    return people_df


def generate_continuous_trajectory(
    pois: gpd.GeoDataFrame,
    start_idx: int,
    pattern: Sequence[str],
    config: SimulationConfig,
) -> List[int]:
    """Follow the notebook path selection for a single daily archetype."""
    if isinstance(pattern, (str, bytes)):
        raise TypeError("Pattern must be a sequence of category names.")

    if config.category_column not in pois.columns:
        raise ValueError(f"POIs missing '{config.category_column}' column.")
    if "geometry" not in pois.columns:
        raise ValueError("POIs must include a 'geometry' column.")
    if start_idx not in pois.index:
        raise KeyError(f"Start index {start_idx} not found in POIs.")

    trajectory: List[int] = [int(start_idx)]
    current_idx = int(start_idx)

    for target_category in pattern:
        candidates = pois[pois[config.category_column] == target_category]
        if candidates.empty:
            break

        current_geom = pois.at[current_idx, "geometry"]
        distances = candidates.geometry.distance(current_geom).to_numpy(dtype=float)
        weights = np.exp(-distances / max(config.distance_scale, 1.0))

        if config.county_column in pois.columns and config.county_column in candidates.columns:
            current_county = pois.at[current_idx, config.county_column]
            same_mask = (candidates[config.county_column] == current_county).fillna(False)
            weights = np.where(
                same_mask.to_numpy(dtype=bool),
                weights * config.county_weight_multiplier.get("same_county", 1.0),
                weights * config.county_weight_multiplier.get("diff_county", 1.0),
            )

        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0) + config.weight_eps
        total = weights.sum()
        if not np.isfinite(total) or total <= 0:
            break

        probabilities = weights / total
        next_idx = int(config.rng.choice(candidates.index.to_numpy(), p=probabilities))
        trajectory.append(next_idx)
        current_idx = next_idx

    return trajectory


def _choose_daily_action(day: int, status: str, config: SimulationConfig) -> str | None:
    """Mirror the notebook's day-by-day archetype selection."""
    if day <= 2:
        if status == "at_home":
            return str(
                config.rng.choice(
                    np.array(("prepping", "evacuating")),
                    p=config.pre_landfall_probs(),
                )
            )
        return None

    if day <= 4:
        return "sheltering_in_place"

    if status == "at_home":
        return "post_storm_recovery"
    if status == "evacuated":
        return "returning_home"

    return None


def simulate_population(
    people_df: pd.DataFrame,
    pois: gpd.GeoDataFrame,
    days: int,
    config: SimulationConfig | None = None,
) -> Dict[int, List[tuple[int, int]]]:
    """Run the day-by-day simulation and return the per-person trajectories."""
    if days <= 0:
        raise ValueError("days must be a positive integer.")

    config = config or SimulationConfig()
    state = people_df.copy()

    required_columns = {"person_id", "status", "current_location_idx"}
    missing = required_columns.difference(state.columns)
    if missing:
        raise ValueError(f"people_df is missing required columns: {sorted(missing)}")

    if "home_location_idx" not in state.columns:
        state["home_location_idx"] = state["current_location_idx"]

    trajectories: Dict[int, List[tuple[int, int]]] = {
        int(pid): [] for pid in state["person_id"].to_numpy(dtype=int)
    }

    state = state.reset_index(drop=True)

    for day in range(1, days + 1):
        for idx, person in state.iterrows():
            pid = int(person["person_id"])
            current_idx = int(person["current_location_idx"])
            status = str(person["status"])

            action = _choose_daily_action(day, status, config)
            pattern: Sequence[str] | None = None
            if action:
                archetype = config.archetypes.get(action)
                if archetype is None:
                    raise KeyError(f"Archetype '{action}' is not defined in the configuration.")
                pattern = archetype.pattern

            daily_traj: List[int] = []
            if pattern:
                daily_traj = generate_continuous_trajectory(pois, current_idx, pattern, config)

            if not trajectories[pid] or trajectories[pid][-1][1] != current_idx:
                trajectories[pid].append((day, current_idx))

            if daily_traj and len(daily_traj) > 1:
                for poi_idx in daily_traj[1:]:
                    trajectories[pid].append((day, int(poi_idx)))

                new_idx = int(daily_traj[-1])
                state.at[idx, "current_location_idx"] = new_idx

                if action == "evacuating":
                    state.at[idx, "status"] = "evacuated"
                elif action == "returning_home" and int(state.at[idx, "home_location_idx"]) == new_idx:
                    state.at[idx, "status"] = "at_home"

    return trajectories


def build_demo_simulation_inputs(
    config: SimulationConfig | None = None,
    n_people: int = 10,
) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """Create a lightweight demo dataset mirroring the notebook schema."""
    config = config or SimulationConfig()

    demo_pois = [
        {
            "poi_id": "demo_home",
            config.category_column: config.home_category,
            config.county_column: "Lee",
            "geometry": Point(-82.462, 26.958),
        },
        {
            "poi_id": "demo_grocery",
            config.category_column: "Grocery Stores",
            config.county_column: "Lee",
            "geometry": Point(-82.458, 26.961),
        },
        {
            "poi_id": "demo_gas",
            config.category_column: "Gasoline Stations",
            config.county_column: "Lee",
            "geometry": Point(-82.455, 26.964),
        },
        {
            "poi_id": "demo_materials",
            config.category_column: "Building Material and Supplies Dealers",
            config.county_column: "Lee",
            "geometry": Point(-82.452, 26.967),
        },
        {
            "poi_id": "demo_hotel",
            config.category_column: "Traveler Accommodation",
            config.county_column: "Lee",
            "geometry": Point(-82.449, 26.97),
        },
        {
            "poi_id": "demo_hospital",
            config.category_column: "Hospitals",
            config.county_column: "Lee",
            "geometry": Point(-82.446, 26.973),
        },
    ]

    pois = gpd.GeoDataFrame(demo_pois, geometry="geometry", crs="EPSG:4326")
    population = {"Lee": 750_000}
    people_df = sample_population(pois, population, n_people, config=config)
    return people_df, pois


def run_simulation(
    people_df: pd.DataFrame,
    pois: gpd.GeoDataFrame,
    days: int | None = None,
    config: SimulationConfig | None = None,
) -> Dict[int, List[tuple[int, int]]]:
    """Convenience wrapper mirroring the notebook entry-point."""
    config = config or SimulationConfig()
    total_days = config.total_days if days is None else days
    return simulate_population(people_df, pois, total_days, config)
