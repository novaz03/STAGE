# Auto-generated from notebook; edit as Python module if needed.
# You can refactor functions into the package (evacmob) and import them in scripts/.

# %% [code] In[1]
import geopandas as gpd
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import folium
import os
import osmnx as ox
import networkx as nx

# %% [code] In[2]
gdf = gpd.read_parquet("POI_encoded_embeddings.parquet")
# print(gdf.head())

# use a UTM projection suitable for Florida (EPSG:5070)
print(f"Original CRS: {gdf.crs}")
gdf = gdf.to_crs("EPSG:5070")
print(f"New CRS: {gdf.crs}. Units are now in meters.")


# %% [code] In[3]
# --- 1. Data Loading & Preprocessing ---
try:
    gdf = gpd.read_parquet("POI_encoded_embeddings.parquet")
    print("✅ Successfully loaded 'POI_encoded_embeddings.parquet'.")
except FileNotFoundError:
    print("❌ ERROR: 'POI_encoded_embeddings.parquet' not found.")
    exit()

gdf = gdf.reset_index(drop=True)

# --- 2. CRS Transformation & Geographic Data Integration ---
TARGET_CRS = "5070"
print("Projecting coordinates...")
gdf_proj = gdf.to_crs(TARGET_CRS) # Create a projected copy
print(f"CRS transformed to Albers Equal Area (meters).")

# --- 3. NEW: Load Population and Hurricane Impact Zone Data ---
print("\nLoading Florida county boundaries and population data...")
try:
    # Load all US county boundaries
    all_counties_gdf = gpd.read_file("https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json")
    
    # filter for Florida counties ONLY
    # The 'STATE' column contains the state FIPS code. Florida's is '12'.
    fl_counties_gdf = all_counties_gdf[all_counties_gdf['STATE'] == '12'].copy()
    
    # Load Florida county population data
    # Ensure this file is in the same directory as your script
    population_df = pd.read_csv("florida_counties_population.csv") # Assumes 'County' and 'Population' columns
    
    # Merge population data into the county geodataframe
    fl_counties_gdf = fl_counties_gdf.merge(population_df, left_on='NAME', right_on='County', how='left')
# Define counties most impacted by Hurricane Ian (names must match data)
    HURRICANE_IAN_COUNTIES = [
        'Lee', 'Charlotte', 'Sarasota', 'Collier', 'DeSoto', 
        'Hardee', 'Manatee', 'Hillsborough', 'Pinellas'
    ]
    
    # Filter for the specific impacted counties within Florida
    ian_zone_gdf = fl_counties_gdf[fl_counties_gdf['County'].isin(HURRICANE_IAN_COUNTIES)].copy().to_crs(5070)
    
    if ian_zone_gdf.empty:
        raise ValueError("Could not find any specified Hurricane Ian counties in the county data. Check names in HURRICANE_IAN_COUNTIES list.")
        
    print(f"✅ Successfully loaded and filtered {len(ian_zone_gdf)} counties in the Hurricane Ian impact zone.")

except Exception as e:
    print(f"❌ Failed to load external geographic or population data: {e}. Cannot continue.")
    exit()

# %% [code] In[4]
# ============================================================
# Hurricane mobility simulator (multi-step/day, 4 cohorts)
# ============================================================
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm.auto import tqdm

# ---- Tunables (use existing ones if already defined) -------
#N_PEOPLE            = globals().get("N_PEOPLE", 150)
N_PEOPLE            = 150
SIMULATION_DAYS     = globals().get("SIMULATION_DAYS", 7)
DISTANCE_SCALE      = globals().get("DISTANCE_SCALE", 20000)  # meters
COUNTY_WEIGHT_MULTIPLIER = globals().get(
    "COUNTY_WEIGHT_MULTIPLIER",
    {"same_county": 1.0, "diff_county": 3.0}   # larger -> more likely
)
RNG = np.random.default_rng(42)

# ---- Preconditions -----------------------------------------
assert 'gdf_proj' in globals(),  "gdf_proj must exist (POIs with geometry, cat_name)"
assert 'ian_zone_gdf' in globals(), "ian_zone_gdf must exist (County, geometry)"

# ---- CRS harmonization (project to a metric CRS) -----------
def pick_local_metric_crs(gdf_ll_4326):
    c = gdf_ll_4326.to_crs(4326).unary_union.centroid
    lon, lat = float(c.x), float(c.y)
    zone = int((lon + 180)//6) + 1
    return f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}"

if gdf_proj.crs is None:
    gdf_proj = gdf_proj.set_crs(4326)
metric_crs = gdf_proj.crs
if metric_crs is None or metric_crs.to_epsg() == 4326:
    metric_crs = pick_local_metric_crs(gdf_proj)
    gdf_proj = gdf_proj.to_crs(metric_crs)
if ian_zone_gdf.crs is None or ian_zone_gdf.crs != metric_crs:
    ian_zone_gdf = ian_zone_gdf.to_crs(metric_crs)

# ---- Join POIs with influenced counties --------------------
pois_all = gpd.sjoin(
    gdf_proj,
    ian_zone_gdf[['County', 'geometry']],
    how="left",
    predicate="intersects"
).drop(columns=["index_right"], errors="ignore")

# Keep legacy alias so downstream code that uses gdf_proj still works
gdf_proj = pois_all

pois_in_zone  = pois_all[pois_all['County'].notna()].copy()
pois_out_zone = pois_all[pois_all['County'].isna()].copy()
if pois_in_zone.empty:
    raise ValueError("No POIs intersect the influenced counties.")

# ---- Category constants ------------------------------------
CAT_HOME  = "Activities Related to Real Estate"
CAT_GROC  = "Grocery Stores"
CAT_GAS   = "Gasoline Stations"
CAT_BUILD = "Building Material and Supplies Dealers"
CAT_HOSP  = "Hospitals"
CAT_HOTL  = "Traveler Accommodation"

# extra categories you asked to include
REST_CATS   = [
    "Restaurants and Other Eating Places",
    "Drinking Places (Alcoholic Beverages)",
    "Special Food Services",
]
RETAIL_CATS = [
    "General Merchandise Stores, including Warehouse Clubs and Supercenters",
    "Grocery Stores", "Health and Personal Care Stores",
    "Electronics and Appliance Stores", "Clothing Stores",
    "Department Stores", "Home Furnishings Stores",
    "Sporting Goods, Hobby, and Musical Instrument Stores",
    "Jewelry, Luggage, and Leather Goods Stores", "Shoe Stores",
    "Book Stores and News Dealers", "Automotive Parts, Accessories, and Tire Stores",
    "Other Miscellaneous Store Retailers", "Furniture Stores",
    "Beer, Wine, and Liquor Stores",
]
TRANSIT_CATS = [
    "Urban Transit Systems",
    "Transit and Ground Passenger Transportation",
    "School and Employee Bus Transportation",
    "Taxi and Limousine Service",
]

# ---- Home pool & assignment --------------------------------
home_locations = pois_in_zone[pois_in_zone['cat_name'] == CAT_HOME]
if home_locations.empty:
    raise ValueError(f"No '{CAT_HOME}' POIs inside the impact zone.")

if 'Population' in ian_zone_gdf.columns:
    county_probs = (ian_zone_gdf.set_index('County')['Population'] /
                    ian_zone_gdf['Population'].sum())
else:
    county_probs = pd.Series(1, index=ian_zone_gdf['County']).astype(float)
    county_probs /= county_probs.sum()

initial_locations = []
for _ in range(N_PEOPLE):
    chosen_cty = RNG.choice(county_probs.index.to_list(), p=county_probs.values)
    homes_in_cty = home_locations[home_locations['County'] == chosen_cty]
    if homes_in_cty.empty:
        initial_locations.append(int(RNG.choice(home_locations.index.to_numpy())))
    else:
        initial_locations.append(int(RNG.choice(homes_in_cty.index.to_numpy())))

people_df = pd.DataFrame({
    "person_id": range(N_PEOPLE),
    "status": "at_home",
    "current_location_idx": initial_locations
})
people_df["home_location_idx"] = people_df["current_location_idx"]
people_df["home_county"]       = people_df["home_location_idx"].map(pois_all["County"])
print(f"✅ Assigned homes to {N_PEOPLE} individuals.")

# ---- Four clearly distinct cohorts -------------------------
# ---- Three clearly distinct cohorts ------------------------
# Split N_PEOPLE as evenly as possible across 3 cohorts
base = N_PEOPLE // 3
rem  = N_PEOPLE % 3
n_sip_home = base + (1 if rem > 0 else 0)
n_sip_hosp = base + (1 if rem > 1 else 0)
n_evac_out = N_PEOPLE - (n_sip_home + n_sip_hosp)

labels = (["sip_home_grocery"] * n_sip_home +
          ["sip_hospital"]     * n_sip_hosp +
          ["evac_out_of_zone"] * n_evac_out)

# randomize assignment but keep index order
labels = pd.Series(labels, index=RNG.permutation(N_PEOPLE)).sort_index().values
people_df["traj_cluster"] = labels
print(people_df["traj_cluster"].value_counts())


# ------------------------------------------------------------
# Picking utilities
# ------------------------------------------------------------
def _distance_weights(cand_gdf: gpd.GeoDataFrame, ref_geom, home_county=None,
                      force_same=False, force_diff=False) -> np.ndarray:
    """Exponential distance kernel; apply optional same/diff-county multipliers."""
    d = cand_gdf.geometry.distance(ref_geom)
    w = np.exp(-d / DISTANCE_SCALE)

    if home_county is not None:
        if force_same:
            mask = cand_gdf["County"].eq(home_county)
            w = w.where(mask, 0.0)
        elif force_diff:
            mask = cand_gdf["County"] != home_county  # True for NaN as well
            w = w.where(mask, 0.0)
        else:
            # soft bias by county
            mask = cand_gdf["County"].eq(home_county)
            w = w * np.where(mask, COUNTY_WEIGHT_MULTIPLIER["same_county"],
                                   COUNTY_WEIGHT_MULTIPLIER["diff_county"])

    w = w.to_numpy()
    w = np.clip(w, 0, None) + 1e-12
    return w / w.sum()

def _sample_idx(cand: gpd.GeoDataFrame, ref_idx: int,
                home_county=None, force_same=False, force_diff=False) -> int | None:
    if cand.empty:
        return None
    ref_geom = pois_all.loc[int(ref_idx), "geometry"]
    probs = _distance_weights(cand, ref_geom, home_county, force_same, force_diff)
    return int(RNG.choice(cand.index.to_numpy(), p=probs))

def _filter_zone(base: gpd.GeoDataFrame, zone: str) -> gpd.GeoDataFrame:
    if zone == "in":
        return base[base["County"].notna()]
    if zone == "out":
        return base[base["County"].isna()]
    return base

def pick_poi(ref_idx: int, category: str,
             zone: str = "any",
             must_same=False, must_diff=False,
             home_county=None) -> int:
    base = _filter_zone(pois_all, zone)
    cand = base[base["cat_name"] == category]
    idx = _sample_idx(cand, ref_idx, home_county, must_same, must_diff)

    # graceful degradations if necessary
    if idx is None and zone != "any":
        idx = _sample_idx(pois_all[pois_all["cat_name"] == category],
                          ref_idx, home_county, must_same, must_diff)
    if idx is None:
        idx = _sample_idx(base, ref_idx, home_county, must_same, must_diff)
    return int(idx) if idx is not None else int(ref_idx)

def pick_from_any(ref_idx: int, categories: list[str],
                  zone: str = "any",
                  must_same=False, must_diff=False,
                  home_county=None) -> int:
    base = _filter_zone(pois_all, zone)
    cand = base[base["cat_name"].isin(categories)]
    idx = _sample_idx(cand, ref_idx, home_county, must_same, must_diff)
    if idx is None and zone != "any":
        cand2 = pois_all[pois_all["cat_name"].isin(categories)]
        idx = _sample_idx(cand2, ref_idx, home_county, must_same, must_diff)
    if idx is None:
        idx = _sample_idx(base, ref_idx, home_county, must_same, must_diff)
    return int(idx) if idx is not None else int(ref_idx)

def ensure_min_errands(steps_today: list[int], min_errands: int,
                       zone: str, home_cty) -> None:
    """
    Ensure at least `min_errands` movements excluding start.
    Called only on non-landfall days.
    """
    need = max(0, (1 + min_errands) - len(steps_today))  # start + min_errands
    for _ in range(need):
        last = steps_today[-1]
        # try restaurants → retail → gas/grocery → transit (in that order)
        for cats in [REST_CATS, RETAIL_CATS, [CAT_GAS, CAT_GROC], TRANSIT_CATS]:
            # only use categories that exist at all
            available = [c for c in cats if c in pois_all["cat_name"].values]
            if not available:
                continue
            nxt = pick_from_any(last, available, zone=zone, home_county=home_cty)
            if nxt != last:
                steps_today.append(nxt)
                break

def return_home(current_idx: int, home_idx: int) -> int:
    return int(home_idx)

# ---- Landfall setting --------------------------------------
LANDFALL_DAYS = {2, 3}  # hurricane hits on days 2–3

def is_in_zone(poi_idx: int) -> bool:
    return pd.notna(pois_all.at[int(poi_idx), "County"])

# ------------------------------------------------------------
# Daily playbooks (multi-step; explicit end-of-day state)
# ------------------------------------------------------------
def simulate_person(pid: int, start_idx: int, cluster: str, days: int) -> list[tuple[int, int]]:
    """
    Returns list of (day, poi_index) including within-day moves,
    and an explicit final step each day (home or hotel) by cohort.
    """
    path: list[tuple[int, int]] = []
    idx  = int(start_idx)
    home = int(start_idx)
    home_cty = people_df.loc[pid, "home_county"]

    for day in range(1, days + 1):
        steps_today: list[int] = [idx]  # start-of-day position
        in_zone_now = is_in_zone(idx)
        on_landfall_in_zone = (day in LANDFALL_DAYS) and in_zone_now

        if cluster == "sip_home_grocery":
            if day == 1:
                steps_today.append(pick_poi(idx, CAT_GROC, zone="in", home_county=home_cty))
                steps_today.append(pick_from_any(steps_today[-1], REST_CATS or [CAT_GAS], zone="in", home_county=home_cty))
                ensure_min_errands(steps_today, 2, zone="in", home_cty=home_cty)
            elif on_landfall_in_zone:
                # stay home; no errands
                pass
            else:
                steps_today.append(pick_poi(idx, CAT_BUILD, zone="in", must_same=True, home_county=home_cty))
                steps_today.append(pick_from_any(steps_today[-1], RETAIL_CATS or REST_CATS, zone="in", home_county=home_cty))
                ensure_min_errands(steps_today, 2, zone="in", home_cty=home_cty)
            steps_today.append(return_home(steps_today[-1], home))

        elif cluster == "sip_hospital":
            if day == 1:
                steps_today.append(pick_from_any(idx, [CAT_GROC] + REST_CATS, zone="in", home_county=home_cty))
                ensure_min_errands(steps_today, 2, zone="in", home_cty=home_cty)
            elif on_landfall_in_zone:
                steps_today.append(pick_poi(idx, CAT_HOSP, zone="in", must_same=True, home_county=home_cty))
                # no padding on landfall
            else:
                steps_today.append(pick_poi(idx, CAT_BUILD, zone="in", must_same=True, home_county=home_cty))
                steps_today.append(pick_from_any(steps_today[-1], REST_CATS or RETAIL_CATS, zone="in", home_county=home_cty))
                ensure_min_errands(steps_today, 2, zone="in", home_cty=home_cty)
            steps_today.append(return_home(steps_today[-1], home))

        elif cluster == "evac_out_of_zone":
            if day == 1:
                steps_today.append(pick_poi(idx, CAT_GROC, zone="in", home_county=home_cty))
                steps_today.append(pick_from_any(steps_today[-1], [CAT_GAS] + (REST_CATS or []), zone="in", home_county=home_cty))
                ensure_min_errands(steps_today, 2, zone="in", home_cty=home_cty)
                steps_today.append(return_home(steps_today[-1], home))
            elif day == 2:
                steps_today.append(pick_poi(idx, CAT_GAS, zone="in", home_county=home_cty))
                steps_today.append(pick_poi(steps_today[-1], CAT_HOTL, zone="out", must_diff=True, home_county=home_cty))
            elif day in {3, 4, 5}:
                # outside zone; errands allowed
                steps_today.append(pick_from_any(steps_today[-1], REST_CATS + RETAIL_CATS, zone="out", home_county=home_cty))
                ensure_min_errands(steps_today, 2, zone="out", home_cty=home_cty)
                steps_today.append(pick_poi(steps_today[-1], CAT_HOTL, zone="out", home_county=home_cty))
            elif day == 6:
                steps_today.append(pick_from_any(steps_today[-1], TRANSIT_CATS or [CAT_GAS, CAT_GROC], zone="in", home_county=home_cty))
                ensure_min_errands(steps_today, 2, zone="in", home_cty=home_cty)
                steps_today.append(pick_poi(steps_today[-1], CAT_HOTL, zone="in", home_county=home_cty))
            else:  # day 7
                steps_today.append(return_home(steps_today[-1], home))
                steps_today.append(pick_poi(steps_today[-1], CAT_BUILD, zone="in", must_same=True, home_county=home_cty))
                ensure_min_errands(steps_today, 2, zone="in", home_cty=home_cty)
                steps_today.append(return_home(steps_today[-1], home))


        # compact consecutive duplicates but keep last
        compact = [steps_today[0]]
        for s in steps_today[1:]:
            if s != compact[-1]:
                compact.append(s)

        # write rows
        for poi_idx in compact:
            path.append((day, int(poi_idx)))

        idx = int(compact[-1])  # next day's start

    return path

# ---- Run simulation ----------------------------------------------------------
print(f"\nStarting {SIMULATION_DAYS}-day simulation…")
full_trajectories = {}
for _, person in tqdm(people_df.iterrows(), total=len(people_df), desc="Simulating"):
    pid       = int(person["person_id"])
    start_idx = int(person["current_location_idx"])
    cluster   = person["traj_cluster"]
    full_trajectories[pid] = simulate_person(pid, start_idx, cluster, SIMULATION_DAYS)
print("✅ Simulation complete.")

# ---- Export (compatible with your pipeline) ---------------------------------
print("\nExporting trajectories to CSV (with cluster + is_home)…")
gdf_proj_wgs84 = gdf_proj.to_crs(4326)
pid_to_cluster  = people_df.set_index("person_id")["traj_cluster"]
pid_to_home_idx = people_df.set_index("person_id")["home_location_idx"]

rows = []
for pid, path in full_trajectories.items():
    cluster = pid_to_cluster.loc[pid]
    homeidx = int(pid_to_home_idx.loc[pid])
    for step_order, (day, poi_idx) in enumerate(path, start=1):
        poi     = gdf_proj.loc[poi_idx]
        poi_wgs = gdf_proj_wgs84.loc[poi_idx]
        poi_name = (
            poi.get("poi_name")
            if "poi_name" in poi
            else (poi.get("label_pair") if "label_pair" in poi else None)
        )
        rows.append({
            "person_id": pid,
            "traj_cluster": cluster,
            "day": day,
            "step_order": step_order,
            "poi_index": int(poi_idx),
            "poi_name": poi_name,
            "poi_category": poi.get("cat_name"),
            "county": poi.get("County"),
            "latitude": float(poi_wgs.geometry.y),
            "longitude": float(poi_wgs.geometry.x),
            "home_location_idx": homeidx,
            "is_home": bool(poi_idx == homeidx),
        })

trajectories_df = pd.DataFrame(rows)
trajectories_df.to_csv("hurricane_ian_trajectories.csv", index=False)
print("✅ Saved hurricane_ian_trajectories.csv")
print(trajectories_df.head(20))

# ---- Quick checks ------------------------------------------------------------
visited_hospital = (trajectories_df["poi_category"] == CAT_HOSP) \
                    .groupby(trajectories_df["person_id"]).any()
print("People who visited hospitals:", int(visited_hospital.sum()))
print(trajectories_df["traj_cluster"].value_counts())

# %% [code] In[5]
# --- 7. Visualization ---
print("\nGenerating interactive GeoJSON highlight map...")
gdf_viz = gdf.to_crs("EPSG:4326")
map_center = [gdf_viz.geometry.y.mean(), gdf_viz.geometry.x.mean()]
features = []
POI_NAME_COLUMN = 'label_pair' 
for person_id, path_with_days in full_trajectories.items():
    if len(path_with_days) > 1:
        points = [(gdf_viz.loc[idx, 'geometry'].x, gdf_viz.loc[idx, 'geometry'].y) for day, idx in path_with_days]
        popup_html = f"<b>Trajectory for Person {person_id}</b><br><hr>"
        popup_html += "<table><tr><th>Day</th><th>Step</th><th>Location Name</th></tr>"
        for i, (day, poi_index) in enumerate(path_with_days):
            poi_name = gdf.loc[poi_index, POI_NAME_COLUMN]
            popup_html += f"<tr><td>{day}</td><td>{i+1}</td><td>{poi_name}</td></tr>"
        popup_html += "</table>"
        feature = {"type": "Feature","geometry": {"type": "LineString", "coordinates": points},"properties": {"person_id": f"Person {person_id}","popup_details": popup_html}}
        features.append(feature)
geojson_data = {"type": "FeatureCollection", "features": features}
m = folium.Map(location=map_center, zoom_start=8, tiles="cartodbpositron")
style_function = lambda x: {"color": "gray", "weight": 1.5, "opacity": 0.5}
highlight_function = lambda x: {"color": "#0077ff", "weight": 4, "opacity": 1.0}
interactive_layer = folium.GeoJson(
    geojson_data,
    style_function=style_function,
    highlight_function=highlight_function,
    tooltip=folium.GeoJsonPopup(fields=['person_id'], aliases=['']),
    popup=folium.GeoJsonPopup(fields=['popup_details'], aliases=[''])
)
m.add_child(interactive_layer)
output_filename = os.path.join("ian_zone_interactive_map.html")
m.save(output_filename)
print(f"🗺️  Interactive map based on Hurricane Ian impact zone and population saved to '{output_filename}'")

# ---- Export (compatible with your pipeline) ---------------------------------
print("\nExporting trajectories to CSV (with cluster + is_home)…")
gdf_proj_wgs84 = gdf_proj.to_crs(4326)
pid_to_cluster  = people_df.set_index("person_id")["traj_cluster"]
pid_to_home_idx = people_df.set_index("person_id")["home_location_idx"]

rows = []
for pid, path in full_trajectories.items():
    cluster = pid_to_cluster.loc[pid]
    homeidx = int(pid_to_home_idx.loc[pid])
    for step_order, (day, poi_idx) in enumerate(path, start=1):
        poi     = gdf_proj.loc[poi_idx]
        poi_wgs = gdf_proj_wgs84.loc[poi_idx]
        poi_name = (
            poi.get("poi_name")
            if "poi_name" in poi
            else (poi.get("label_pair") if "label_pair" in poi else None)
        )
        rows.append({
            "person_id": pid,
            "traj_cluster": cluster,
            "day": day,
            "step_order": step_order,
            "poi_index": int(poi_idx),
            "poi_name": poi_name,
            "poi_category": poi.get("cat_name"),
            "county": poi.get("County"),
            "latitude": float(poi_wgs.geometry.y),
            "longitude": float(poi_wgs.geometry.x),
            "home_location_idx": homeidx,
            "is_home": bool(poi_idx == homeidx),
        })

trajectories_df = pd.DataFrame(rows)
trajectories_df.to_csv("hurricane_ian_trajectories.csv", index=False)
print("✅ Saved hurricane_ian_trajectories.csv")
print(trajectories_df.head(20))

# ---- Quick checks ------------------------------------------------------------
visited_hospital = (trajectories_df["poi_category"] == CAT_HOSP) \
                    .groupby(trajectories_df["person_id"]).any()
print("People who visited hospitals:", int(visited_hospital.sum()))
print(trajectories_df["traj_cluster"].value_counts())

# %% [code] In[6]
import pandas as pd
import numpy as np
import geopandas as gpd
import osmnx as ox

# Ensure POIs are in WGS84 for distance calcs
pois_wgs = gdf_proj.to_crs(4326) if (gdf_proj.crs and gdf_proj.crs.to_epsg() != 4326) else gdf_proj

# --------- Classifier for the 4 cohorts --------------------------------------
def classify_traj_4(features):
    visited_traveler      = features["visited_traveler"]
    visited_hospital      = features["visited_hospital"]
    left_home_county      = features["left_home_county"]       # True if any County != home_county
    touched_outside_zone  = features["touched_outside_zone"]   # True if any County is NaN
    returned_home_end     = features["returned_home_end"]      # ended at home idx
    n_moves               = features["n_moves"]
    n_unique_pois         = features["n_unique_pois"]

    # Degenerate: no movement at all → treat as shelter at home
    if n_moves == 0 or n_unique_pois <= 1:
        return "sip_home_grocery", "No movement; stayed at or near home."

    # Evacuation with hotel OUT of influenced zone (County==NaN at least once)
    if visited_traveler and touched_outside_zone:
        return "evac_out_of_zone", "Traveler accommodation outside influenced counties."

    # Short evacuation WITHIN influenced zone (traveler accommodation, no NaN County)
    if visited_traveler and not touched_outside_zone:
        return "evac_short_in_zone", "Traveler accommodation but stayed within influenced counties."

    # Shelter at hospital (no traveler accommodation; in-zone only by construction)
    if visited_hospital and not visited_traveler and not touched_outside_zone:
        return "sip_hospital", "Visited hospital; did not use traveler accommodation or leave the zone."

    # Default to shelter at home/grocery if no hotel/hospital signal and stayed in-zone
    if (not visited_traveler) and (not visited_hospital) and (not touched_outside_zone):
        return "sip_home_grocery", "In-zone errands only (home/groceries/etc.)."

    # Fallback: map anything else to the closest cohort
    if visited_hospital:
        return "sip_hospital", "Hospital visit dominates classification."
    if visited_traveler:
        # If we couldn’t tell zone, use short-in-zone as neutral fallback
        return "evac_short_in_zone", "Traveler accommodation present; zone ambiguous."
    return "sip_home_grocery", "No evac/hospital; default to shelter-at-home."
# -----------------------------------------------------------------------------

rows = []
for person_id, path in full_trajectories.items():
    if not path:
        continue

    poi_idxs = [idx for (_, idx) in path if idx in pois_wgs.index]
    if not poi_idxs:
        continue

    # Categories visited
    cats_series = pois_wgs.loc[poi_idxs, 'cat_name'].fillna("")
    cats_set = set(cats_series.tolist())
    visited_traveler = ("Traveler Accommodation" in cats_set)
    visited_hospital = ("Hospitals" in cats_set)

    # Home info
    home_idx = int(people_df.loc[people_df['person_id'] == person_id, 'home_location_idx'].iloc[0])
    home_geom = pois_wgs.geometry[home_idx]
    home_lat, home_lon = float(home_geom.y), float(home_geom.x)
    home_county = pois_wgs.loc[home_idx, 'County'] if 'County' in pois_wgs.columns else np.nan

    # Destination coords
    dest_geoms = pois_wgs.geometry.loc[poi_idxs]
    lats = dest_geoms.y.to_numpy()
    lons = dest_geoms.x.to_numpy()

    # Great-circle distances home -> each stop
    if len(lats) > 0:
        dists_m = ox.distance.great_circle(
            np.full_like(lats, home_lat, dtype=float),
            np.full_like(lats, home_lon, dtype=float),
            lats, lons
        )
        max_dist_km = float(np.nanmax(dists_m) / 1000.0)
    else:
        max_dist_km = 0.0

    # County movement features
    counties = pois_wgs.loc[poi_idxs, 'County'] if 'County' in pois_wgs.columns else pd.Series([np.nan] * len(poi_idxs))
    left_home_county = any((pd.notna(c) and (c != home_county)) for c in counties)
    touched_outside_zone = any(pd.isna(c) for c in counties)  # NaN => outside influenced counties

    # Returned home by END (strict)
    end_idx = poi_idxs[-1]
    returned_home_end = (end_idx == home_idx)

    n_unique_pois = len(set(poi_idxs))
    n_moves = max(0, len(poi_idxs) - 1)

    feats = dict(
        person_id=person_id,
        n_moves=n_moves,
        n_unique_pois=n_unique_pois,
        cats=cats_set,
        visited_traveler=visited_traveler,
        visited_hospital=visited_hospital,
        left_home_county=left_home_county,
        touched_outside_zone=touched_outside_zone,
        returned_home_end=returned_home_end,
        max_dist_km=max_dist_km,
        home_idx=home_idx,
        end_idx=end_idx,
        home_county=home_county,
        end_county=pois_wgs.loc[end_idx, 'County'] if 'County' in pois_wgs.columns else np.nan,
    )
    label = classify_traj_4(feats)
    # If classify_traj_4 returns a tuple from an older version, handle that too
    if isinstance(label, tuple):
        label, reason = label
    else:
        reason = ""
    feats["traj_cluster"] = label
    feats["cluster_reason"] = reason
    rows.append(feats)

# Person-level summary with labels
traj_summary_df = pd.DataFrame(rows).sort_values(["traj_cluster", "person_id"]).reset_index(drop=True)

# Attach label to step-level trajectories_df (if present)
if 'trajectories_df' in locals():
    trajectories_df = trajectories_df.merge(
        traj_summary_df[["person_id", "traj_cluster"]],
        on="person_id",
        how="left"
    )

# Also attach to people_df
people_df = people_df.merge(
    traj_summary_df[["person_id", "traj_cluster"]],
    on="person_id",
    how="left"
)

print("\nTrajectory clusters assigned. Summary preview:")
print(traj_summary_df["traj_cluster"].value_counts(dropna=False))
print(traj_summary_df.head())

# Save (optional)
os.makedirs("./outputs", exist_ok=True)
traj_summary_df.to_csv("./outputs/trajectory_clusters_summary.csv", index=False)
if 'trajectories_df' in locals():
    trajectories_df.to_csv("./outputs/hurricane_ian_trajectories_with_cluster.csv", index=False)

# %% [code] In[7]
import os, random
import folium
import pandas as pd
import geopandas as gpd

# ---------- settings ----------
OUTPUT_DIR = locals().get("output_dir", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_HTML = os.path.join(OUTPUT_DIR, "simulated_trajectories.html")
MAX_PEOPLE_ON_MAP = 99    # change to None or a large number to show all

# ---------- data checks / CRS ----------
assert 'full_trajectories' in locals(), "full_trajectories not found."
assert 'gdf_proj' in locals(), "gdf_proj (POIs) not found."

# Map clusters for people (prefer people_df; else fall back to trajectories_df; else 'unknown')
cluster_map = {}
if 'people_df' in locals() and 'traj_cluster' in people_df.columns:
    cluster_map = pd.Series(people_df['traj_cluster'].values,
                            index=people_df['person_id']).to_dict()
elif 'trajectories_df' in locals() and ('traj_cluster_y' in trajectories_df.columns or 'traj_cluster' in trajectories_df.columns):
    try:
        cluster_map = trajectories_df.groupby('person_id')['traj_cluster_y'].first().to_dict()
    except:
        cluster_map = trajectories_df.groupby('person_id')['traj_cluster'].first().to_dict()

else:
    # leave empty; we'll default to "unknown" below
    pass

# Optional: consistent colors per cluster (fallback palette used if label is unknown)
CLUSTER_COLORS = {
    'sip_home_grocery':   '#1f77b4',  # blue
    'sip_hospital':       '#d62728',  # red
    'evac_out_of_zone':   '#9467bd',  # purple
    'evac_short_in_zone': '#2ca02c',  # green
    'unknown':            '#7f7f7f',  # gray
}

# Folium expects WGS84 for coordinates
pois_wgs = gdf_proj.to_crs(4326) if (gdf_proj.crs and gdf_proj.crs.to_epsg() != 4326) else gdf_proj

# Map center
if len(pois_wgs) and pois_wgs.geometry.notnull().any():
    c = pois_wgs.unary_union.centroid
    map_center = [c.y, c.x]
else:
    map_center = [27.8, -81.7]  # fallback: central Florida-ish

m = folium.Map(location=map_center, zoom_start=8, tiles="CartoDB Positron")

# Which people to show
person_ids = sorted(pid for pid, traj in full_trajectories.items() if traj)
if MAX_PEOPLE_ON_MAP is not None and len(person_ids) > MAX_PEOPLE_ON_MAP:
    person_ids = random.sample(person_ids, MAX_PEOPLE_ON_MAP)

# fallback palette (used when cluster label unknown)
fallback_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                   '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# track bounds
minx = miny =  1e99
maxx = maxy = -1e99

# POI name column (fallback if missing)
POI_NAME_COLUMN = 'label_pair' if 'label_pair' in pois_wgs.columns else (
                  'name' if 'name' in pois_wgs.columns else None)

for i, pid in enumerate(person_ids):
    traj = full_trajectories[pid]  # list of (day, poi_index)
    if not traj:
        continue

    cluster = cluster_map.get(pid, 'unknown')
    color = CLUSTER_COLORS.get(cluster, fallback_colors[i % len(fallback_colors)])

    # Collect this person's coordinates and step info
    coords = []            # [[lat, lon], ...] for polyline
    step_rows = []         # dicts for markers
    for step_idx, (day, poi_idx) in enumerate(traj, start=1):
        if poi_idx not in pois_wgs.index:
            continue
        geom = pois_wgs.loc[poi_idx, 'geometry']
        lat, lon = float(geom.y), float(geom.x)
        poi_name = str(pois_wgs.loc[poi_idx].get(POI_NAME_COLUMN, "POI"))
        coords.append([lat, lon])
        step_rows.append({"step": step_idx, "day": day, "lat": lat, "lon": lon, "name": poi_name})

        # expand bounds
        minx, miny = min(minx, lon), min(miny, lat)
        maxx, maxy = max(maxx, lon), max(maxy, lat)

    if not coords:
        continue

    # Layer is named with person id + cluster
    layer = folium.FeatureGroup(name=f"Person {pid} · {cluster}", show=(i == 0))

    # Trajectory line
    folium.PolyLine(locations=coords, color=color, weight=4, opacity=0.9).add_to(layer)

    # Dots + step labels
    for r in step_rows:
        folium.CircleMarker(
            location=[r['lat'], r['lon']],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=1.0,
            popup=folium.Popup(
                f"<b>Person {pid}</b><br>"
                f"Cluster: <b>{cluster}</b><br>"
                f"Day {r['day']} — Step {r['step']}<br>"
                f"{r['name']}",
                max_width=320
            )
        ).add_to(layer)

        # step number as small text label next to the dot
        folium.Marker(
            location=[r['lat'], r['lon']],
            icon=folium.DivIcon(html=f"""
                <div style="
                    font-size:10px; font-weight:bold; color:{color};
                    text-shadow:-1px -1px 0 #fff, 1px -1px 0 #fff,
                                 -1px  1px 0 #fff, 1px  1px 0 #fff;
                    transform: translate(8px, -8px);
                ">{r['step']}</div>""")
        ).add_to(layer)

    # Mark the START position with a star icon
    start = step_rows[0]
    folium.Marker(
        location=[start['lat'], start['lon']],
        tooltip=f"Person {pid} · {cluster} · START",
        popup=folium.Popup(
            f"<b>Person {pid}</b><br>"
            f"Cluster: <b>{cluster}</b><br>"
            f"START — Day {start['day']} — Step {start['step']}<br>"
            f"{start['name']}",
            max_width=320
        ),
        icon=folium.Icon(icon='star', prefix='fa', color='black', icon_color=color)
    ).add_to(layer)

    layer.add_to(m)

# Fit map to bounds if we added anything
if minx < maxx and miny < maxy:
    m.fit_bounds([[miny, minx], [maxy, maxx]])

folium.LayerControl(collapsed=False).add_to(m)

# Save / show
m.save(OUTPUT_HTML)
print(f"🗺️  Simulated trajectories map saved to: {OUTPUT_HTML}")

try:
    display(m)  # Jupyter
except Exception:
    pass

# %% [code] In[8]
# --- imports (safe if repeated) ---
import os
import logging
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import networkx as nx
import geopandas as gpd
import osmnx as ox
import shapely
from osmnx.routing import route_to_gdf
from shapely.ops import linemerge

# %% [code] In[9]
# --- Build AOI from convex hull (+10 km buffer, no holes) --------------------
import os
import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.ops import unary_union
from shapely.geometry import Polygon
from shapely.validation import make_valid as _make_valid  # may not exist on older shapely

# 0) Collect trajectory points in WGS84
if 'trajectories_df' in locals() and not trajectories_df.empty:
    pts_df = trajectories_df[['longitude', 'latitude']].dropna()
else:
    # fallback: build from full_trajectories + gdf_proj
    assert 'full_trajectories' in locals() and 'gdf_proj' in locals(), \
        "Need trajectories_df or (full_trajectories + gdf_proj)."
    pois_wgs = gdf_proj.to_crs(4326) if (gdf_proj.crs and gdf_proj.crs.to_epsg() != 4326) else gdf_proj
    rows = []
    for _, path in full_trajectories.items():
        for _, poi_idx in path:
            if poi_idx in pois_wgs.index:
                g = pois_wgs.loc[poi_idx, 'geometry']
                rows.append((g.x, g.y))
    pts_df = pd.DataFrame(rows, columns=['longitude', 'latitude']).dropna()

if pts_df.empty:
    raise ValueError("No trajectory points found to build the AOI.")

pts_ll = gpd.GeoDataFrame(geometry=gpd.points_from_xy(pts_df.longitude, pts_df.latitude), crs=4326)

# 1) Project to a *local metric CRS* (UTM) for buffer/distances
def pick_utm_epsg(gdf_ll):
    c = gdf_ll.unary_union.centroid
    lon, lat = float(c.x), float(c.y)
    zone = int((lon + 180) // 6) + 1
    return f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}"

metric_epsg = pick_utm_epsg(pts_ll)
pts_m = pts_ll.to_crs(metric_epsg)

# 2) Convex hull (has no holes by definition), then buffer 10 km
hull_m = pts_m.unary_union.convex_hull           # Polygon/LineString/Point depending on #points
aoi_m = hull_m.buffer(10_000)                    # expand by 10 km

# 3) Ensure valid + drop any interiors just in case
def make_valid(geom):
    try:
        return _make_valid(geom)
    except Exception:
        return gpd.GeoSeries([geom], crs=metric_epsg).buffer(0).iloc[0]

def drop_holes(geom):
    if geom.geom_type == "Polygon":
        return Polygon(geom.exterior)
    if geom.geom_type == "MultiPolygon":
        return unary_union([Polygon(p.exterior) for p in geom.geoms])
    return geom

aoi_m = make_valid(aoi_m)
aoi_m = drop_holes(aoi_m)

# 4) Optional: simplify for a lighter Overpass request (tolerance in meters)
aoi_m = gpd.GeoSeries([aoi_m], crs=metric_epsg).simplify(250, preserve_topology=True).iloc[0]

# 5) Back to WGS84 for OSMnx
aoi_ll = gpd.GeoSeries([aoi_m], crs=metric_epsg).to_crs(4326).iloc[0]

# --- Quick visualization (static) -------------------------------------------
# pip install contextily  # if you don't have it yet

import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as cx

# wrap and project for web tiles
gs_ll = gpd.GeoSeries([aoi_ll], crs=4326)
gs_3857 = gs_ll.to_crs(3857)

fig, ax = plt.subplots(figsize=(7, 7))

# AOI style (outline only; make facecolor='none' for no fill)
gs_3857.plot(ax=ax, facecolor="none", edgecolor="#0077ff", linewidth=2, zorder=3)

# pad the view a bit so the outline isn't glued to edges
minx, miny, maxx, maxy = gs_3857.total_bounds
pad = max((maxx - minx), (maxy - miny)) * 0.05  # 5% padding
ax.set_xlim(minx - pad, maxx + pad)
ax.set_ylim(miny - pad, maxy + pad)

# add basemap
cx.add_basemap(
    ax,
    crs="EPSG:3857",
    source=cx.providers.CartoDB.Positron,  # choose any provider you like
    attribution_size=6
)

ax.set_title("AOI (Convex Hull + 10 km buffer) with Basemap")
ax.set_axis_off()
plt.show()

# optional: save to file
# fig.savefig("aoi_with_basemap.png", dpi=150, bbox_inches="tight")


# --- Download OSM network for this AOI (change network_type if needed) ------
print("\nDownloading OSM network for convex-hull AOI…")
ox.settings.use_cache = True
ox.settings.log_console = True
ox.settings.cache_folder = "./cache"
ox.settings.requests_timeout = 180

try:
    G = ox.graph_from_polygon(
        aoi_ll,
        network_type="drive",    # or "all" for walk/bike too
        simplify=True,
        truncate_by_edge=True,
        retain_all=False
    )
    print("✅ Road network graph downloaded successfully.")
except Exception as e:
    print(f"❌ Polygon download failed. Error: {e}")
    west, south, east, north = gpd.GeoSeries([aoi_ll], crs=4326).total_bounds
    print("→ Trying bounding-box fallback…")
    G = ox.graph_from_bbox(north, south, east, west, network_type="drive", simplify=True)
    print("✅ Road network (bbox) downloaded successfully.")

# --- Save AOI and graph (optional) ------------------------------------------
out_dir = "trajectory_roadmap_convex"
os.makedirs(out_dir, exist_ok=True)
gpd.GeoSeries([aoi_ll], crs=4326).to_file(os.path.join(out_dir, "aoi_convex.geojson"), driver="GeoJSON")
ox.save_graphml(G, os.path.join(out_dir, "roadmap_convex.graphml"))
ox.save_graph_geopackage(G, filepath=os.path.join(out_dir, "roadmap_convex.gpkg"), directed=True)
print(f"📦 Saved AOI to {os.path.join(out_dir, 'aoi_convex.geojson')}")
print(f"📦 Saved GraphML to {os.path.join(out_dir, 'roadmap_convex.graphml')}")
print(f"📦 Saved GeoPackage to {os.path.join(out_dir, 'roadmap_convex.gpkg')}")

# %% [code] In[10]

out_dir = "trajectory_roadmap_convex"

# A) AOI (GeoJSON) -> shapely (Multi)Polygon in WGS84
aoi_gs = gpd.read_file(os.path.join(out_dir, "aoi_convex.geojson"))
aoi_gs = aoi_gs.to_crs(4326)          # ensure WGS84
aoi_ll = aoi_gs.geometry.iloc[0]      # <- you'll have this available if needed later

# B) Graph
graphml_path = os.path.join(out_dir, "roadmap_convex.graphml")
gpkg_path    = os.path.join(out_dir, "roadmap_convex.gpkg")

if os.path.exists(graphml_path):
    # Preferred: GraphML keeps the NetworkX graph structure
    G = ox.load_graphml(graphml_path)                 # MultiDiGraph, nodes have x/y (lon/lat)
else:
    # Fallback: rebuild graph from the GeoPackage layers
    nodes = gpd.read_file(gpkg_path, layer="nodes")
    edges = gpd.read_file(gpkg_path, layer="edges")
    # Ensure WGS84 for consistency with your routing code
    if nodes.crs and nodes.crs.to_epsg() != 4326:
        nodes = nodes.to_crs(4326)
        edges = edges.to_crs(4326)
    G = ox.graph_from_gdfs(nodes, edges)              # MultiDiGraph

# (Optional) coerce node x/y to float if they loaded as strings
for n, d in G.nodes(data=True):
    d["x"] = float(d["x"])
    d["y"] = float(d["y"])

# %% [code] In[11]
# ---------- Build pairs_df from trajectories_df (snapping + next-step joins) ----------
import numpy as np
import pandas as pd
import osmnx as ox

# If needed, load trajectories
if 'trajectories_df' not in locals():
    trajectories_df = pd.read_csv("hurricane_ian_trajectories.csv")

# Ensure expected columns exist
if 'poi_name' not in trajectories_df.columns and 'label_pair' in trajectories_df.columns:
    trajectories_df['poi_name'] = trajectories_df['label_pair']

# Sort once
trajectories_df = trajectories_df.sort_values(['person_id', 'step_order']).reset_index(drop=True)

# Snap each trajectory point to the nearest edge endpoint (use the original MultiDiGraph G here)
Xs = trajectories_df['longitude'].to_numpy()
Ys = trajectories_df['latitude'].to_numpy()
uvks = ox.distance.nearest_edges(G, Xs, Ys)  # [(u,v,k), ...]

u_ids = np.fromiter((t[0] for t in uvks), dtype=object, count=len(uvks))
v_ids = np.fromiter((t[1] for t in uvks), dtype=object, count=len(uvks))

ux = np.fromiter((G.nodes[u]['x'] for u in u_ids), dtype=float,  count=len(u_ids))
uy = np.fromiter((G.nodes[u]['y'] for u in u_ids), dtype=float,  count=len(u_ids))
vx = np.fromiter((G.nodes[v]['x'] for v in v_ids), dtype=float,  count=len(v_ids))
vy = np.fromiter((G.nodes[v]['y'] for v in v_ids), dtype=float,  count=len(v_ids))

# Pick the nearer endpoint (vectorized great-circle; returns meters)
du = ox.distance.great_circle(Ys, Xs, uy, ux)
dv = ox.distance.great_circle(Ys, Xs, vy, vx)
snap_nodes = np.where(du <= dv, u_ids, v_ids)

trajectories_df['snap_node'] = snap_nodes

# Build next-step columns per person
trajectories_df['lon_next']       = trajectories_df.groupby('person_id')['longitude'].shift(-1)
trajectories_df['lat_next']       = trajectories_df.groupby('person_id')['latitude' ].shift(-1)
trajectories_df['dest_poi']       = trajectories_df.groupby('person_id')['poi_name' ].shift(-1)
trajectories_df['snap_node_next'] = trajectories_df.groupby('person_id')['snap_node' ].shift(-1)
trajectories_df['step_next']      = trajectories_df.groupby('person_id')['step_order'].shift(-1)

# Keep only rows that have a following step
pairs_df = trajectories_df.dropna(subset=['snap_node_next']).copy()
pairs_df['snap_node']      = pairs_df['snap_node'].astype(object)
pairs_df['snap_node_next'] = pairs_df['snap_node_next'].astype(object)

# If you already truncated to the largest component, the comp-id filter is optional.
# Otherwise, you can filter unreachable legs like this:
# comp_id = {n: i for i, comp in enumerate(nx.weakly_connected_components(G)) for n in comp}
# pairs_df = pairs_df[pairs_df['snap_node'].map(comp_id) == pairs_df['snap_node_next'].map(comp_id)]

# Drop zero-length legs (origin and destination snap to the same node)
pairs_df = pairs_df[pairs_df['snap_node'] != pairs_df['snap_node_next']].reset_index(drop=True)

print(f"pairs_df ready with {len(pairs_df)} legs from {pairs_df['person_id'].nunique()} people.")

# %% [code] In[12]
from shapely.geometry import LineString
from shapely.ops import linemerge

# -------------------- Imports --------------------
import os
import logging
from typing import Tuple, Iterable, Dict, Any, List

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import networkx as nx
import geopandas as gpd
import osmnx as ox
from osmnx.routing import route_to_gdf
from shapely.ops import linemerge

# -------------------- Config ---------------------
OUT_DIR = "trajectory_roadmap_convex"
AOI_GEOJSON = os.path.join(OUT_DIR, "aoi_convex.geojson")
GRAPHML_PATH = os.path.join(OUT_DIR, "roadmap_convex.graphml")
GPKG_PATH = os.path.join(OUT_DIR, "roadmap_convex.gpkg")
TRAJ_CSV = "hurricane_ian_trajectories.csv"
ROUTES_OUT = "hurricane_ian_routes.parquet"
LOG_FILE = "overlapping.log"


# -------------------- Logging --------------------
def setup_logging() -> None:
    logging.basicConfig(
        filename=LOG_FILE,
        filemode="w",  # overwrite each run; change to "a" to append
        level=logging.INFO,
        format="%(message)s",
    )
    logging.info("Zero-length (overlapping) legs log\n")


# -------------------- IO helpers -----------------
def load_aoi(aoi_path: str) -> gpd.GeoSeries:
    """Load AOI (GeoJSON) and return GeoSeries in EPSG:4326."""
    aoi_gs = gpd.read_file(aoi_path)
    aoi_gs = aoi_gs.to_crs(4326)
    return aoi_gs


def load_graph(graphml_path: str, gpkg_path: str) -> nx.MultiDiGraph:
    """
    Load OSMnx MultiDiGraph from GraphML if present; otherwise rebuild from GPKG nodes/edges.
    Ensures node x/y are floats and CRS is EPSG:4326.
    """
    if os.path.exists(graphml_path):
        G = ox.load_graphml(graphml_path)
    else:
        nodes = gpd.read_file(gpkg_path, layer="nodes")
        edges = gpd.read_file(gpkg_path, layer="edges")
        if nodes.crs and nodes.crs.to_epsg() != 4326:
            nodes = nodes.to_crs(4326)
            edges = edges.to_crs(4326)
        G = ox.graph_from_gdfs(nodes, edges)

    # coerce node coords to float if necessary
    for _, d in G.nodes(data=True):
        d["x"] = float(d["x"])
        d["y"] = float(d["y"])

    print(f"Loaded graph with {G.number_of_nodes():,} nodes / {G.number_of_edges():,} edges")
    return G


# -------------------- Graph prep -----------------
def prep_graph(G: nx.MultiDiGraph) -> Tuple[nx.MultiDiGraph, Dict[Any, int]]:
    """Keep largest weakly-connected component; add speeds & travel times; compute component labels."""
    print("\nStep 3: Preparing graph (largest component, speeds/times)...")

    if G.number_of_edges() == 0 or G.number_of_nodes() == 0:
        raise RuntimeError("The road graph is empty. Re-check the AOI / network_type / download step.")

    # largest component
    G = ox.truncate.largest_component(G, strongly=False)

    # add speeds & times if missing
    if any("speed_kph" not in d for _, _, d in G.edges(data=True)):
        G = ox.add_edge_speeds(G)
    if any("travel_time" not in d for _, _, d in G.edges(data=True)):
        G = ox.add_edge_travel_times(G)

    # component labels for quick reachability checks
    comp_id = {n: i for i, comp in enumerate(nx.weakly_connected_components(G)) for n in comp}

    print("✅ Graph ready.")
    return G, comp_id


# ---------------- Trajectories & snapping --------
def load_trajectories(csv_path: str) -> pd.DataFrame:
    """Load trajectories CSV and sort by person/step."""
    print("\nStep 4: Loading trajectories...")
    df = pd.read_csv(csv_path)
    # Ensure expected name column
    if "poi_name" not in df.columns and "label_pair" in df.columns:
        df["poi_name"] = df["label_pair"]
    df = df.sort_values(["person_id", "step_order"]).reset_index(drop=True)
    return df


def snap_points_to_nodes(G: nx.MultiDiGraph, trajectories_df: pd.DataFrame) -> pd.DataFrame:
    """Snap each trajectory point to the nearest edge endpoint using vectorized ops."""
    print("Snapping all trajectory points to nearest edges (vectorized)...")

    Xs = trajectories_df["longitude"].to_numpy()
    Ys = trajectories_df["latitude"].to_numpy()
    edge_triplets = ox.distance.nearest_edges(G, Xs, Ys)  # [(u,v,key), ...]

    u_ids = np.fromiter((t[0] for t in edge_triplets), dtype=object, count=len(edge_triplets))
    v_ids = np.fromiter((t[1] for t in edge_triplets), dtype=object, count=len(edge_triplets))

    ux = np.fromiter((G.nodes[u]["x"] for u in u_ids), dtype=float, count=len(u_ids))
    uy = np.fromiter((G.nodes[u]["y"] for u in u_ids), dtype=float, count=len(u_ids))
    vx = np.fromiter((G.nodes[v]["x"] for v in v_ids), dtype=float, count=len(v_ids))
    vy = np.fromiter((G.nodes[v]["y"] for v in v_ids), dtype=float, count=len(v_ids))

    # great-circle distances to each candidate endpoint (meters); vectorized
    du = ox.distance.great_circle(Ys, Xs, uy, ux)
    dv = ox.distance.great_circle(Ys, Xs, vy, vx)

    snap_nodes = np.where(du <= dv, u_ids, v_ids)
    trajectories_df = trajectories_df.copy()
    trajectories_df["snap_node"] = snap_nodes
    return trajectories_df
from shapely.geometry import LineString
from shapely.ops import linemerge

def multidi_to_best_digraph(G_multi, weight="travel_time"):
    """Keep one (u,v) edge: the minimum weight. Carry length+geometry."""
    H = nx.DiGraph()
    # copy node coords
    H.add_nodes_from((n, {"x": d["x"], "y": d["y"]}) for n, d in G_multi.nodes(data=True))
    # choose best edge per (u,v)
    for u, v, k, d in G_multi.edges(keys=True, data=True):
        w = d.get(weight)
        if w is None:
            continue
        e = H.get_edge_data(u, v)
        if (e is None) or (w < e.get(weight, float("inf"))):
            H.add_edge(
                u, v,
                travel_time=float(w),
                length=float(d.get("length", 0.0)),
                geometry=d.get("geometry")  # may be None; we’ll synthesize if missing
            )
    return H


def build_pairs_df(G: nx.MultiDiGraph, comp_id: Dict[Any, int], trajectories_df: pd.DataFrame) -> pd.DataFrame:
    """Build leg-level pairs with next-step info; drop unreachable and zero-length legs; log overlaps."""
    print("Building route requests...")

    df = trajectories_df.copy()
    # raw next-step coordinates & POIs
    df["lon_next"] = df.groupby("person_id")["longitude"].shift(-1)
    df["lat_next"] = df.groupby("person_id")["latitude"].shift(-1)
    df["dest_poi"] = df.groupby("person_id")["poi_name"].shift(-1)

    # snapped next nodes
    df["snap_node_next"] = df.groupby("person_id")["snap_node"].shift(-1)
    df["step_next"] = df.groupby("person_id")["step_order"].shift(-1)

    pairs_df = df.dropna(subset=["snap_node_next"]).copy()
    pairs_df["snap_node"] = pairs_df["snap_node"].astype(object)
    pairs_df["snap_node_next"] = pairs_df["snap_node_next"].astype(object)

    # Drop legs in different weak components (unreachable)
    pairs_df = pairs_df[
        pairs_df["snap_node"].map(comp_id) == pairs_df["snap_node_next"].map(comp_id)
    ].copy()

    # Log & drop zero-length (same snapped node) legs
    mask_same = pairs_df["snap_node"] == pairs_df["snap_node_next"]
    zero_len = pairs_df.loc[mask_same].copy()

    def _osmid(n):
        val = G.nodes[n].get("osmid")
        if isinstance(val, (list, tuple)):
            return val[0]
        return val

    if not zero_len.empty:
        zero_len["node_id"] = zero_len["snap_node"]
        zero_len["node_osmid"] = zero_len["snap_node"].map(_osmid)
        zero_len["node_x"] = zero_len["snap_node"].map(lambda n: G.nodes[n]["x"])
        zero_len["node_y"] = zero_len["snap_node"].map(lambda n: G.nodes[n]["y"])

        zero_len["gc_m"] = ox.distance.great_circle(
            zero_len["latitude"].to_numpy(),
            zero_len["longitude"].to_numpy(),
            zero_len["lat_next"].to_numpy(),
            zero_len["lon_next"].to_numpy(),
        )

        cols = [
            "person_id",
            "step_order",
            "poi_name",
            "dest_poi",
            "node_id",
            "node_osmid",
            "node_x",
            "node_y",
            "longitude",
            "latitude",
            "lon_next",
            "lat_next",
            "gc_m",
        ]
        cols = [c for c in cols if c in zero_len.columns]

        logging.info(f"Skipping {len(zero_len)} zero-length legs (same snapped node). Details:")
        for _, r in zero_len[cols].iterrows():
            logging.info(
                "person_id=%s step=%s orig_poi=%s dest_poi=%s "
                "snap_node=%s osmid=%s node_x=%.6f node_y=%.6f "
                "orig_lon=%.6f orig_lat=%.6f dest_lon=%.6f dest_lat=%.6f gc_m=%.2f",
                r.get("person_id"),
                int(r.get("step_order")),
                str(r.get("poi_name")),
                str(r.get("dest_poi")),
                r.get("node_id"),
                str(r.get("node_osmid")),
                float(r.get("node_x")),
                float(r.get("node_y")),
                float(r.get("longitude")),
                float(r.get("latitude")),
                float(r.get("lon_next")),
                float(r.get("lat_next")),
                float(r.get("gc_m")),
            )

    pairs_df = pairs_df[~mask_same].copy()
    print(f"Total legs to route (after filtering): {len(pairs_df)}")
    return pairs_df


# -------------------- Routing --------------------
def route_legs(
    G: nx.MultiDiGraph,
    pairs_df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """Route all legs using a single-source Dijkstra per origin (directed only)."""
    print("Routing with single-source Dijkstra per origin (directed only)...")

    all_routes: List[Dict[str, Any]] = []

    # group by origin snapped node → one Dijkstra per origin
    for o_node, group in tqdm(pairs_df.groupby("snap_node"), total=pairs_df["snap_node"].nunique()):
        # One Dijkstra computes all reachable shortest paths from o_node
        _, paths_dir = nx.single_source_dijkstra(G, source=o_node, weight="travel_time")

        # Emit results for each leg in this group
        for _, row in group.iterrows():
            d_node = row["snap_node_next"]
            route_nodes = paths_dir.get(d_node)  # None if unreachable (unlikely after comp filter)
            if route_nodes is None or len(route_nodes) < 2:
                continue

            edges_gdf = route_to_gdf(G, route_nodes, weight="travel_time")  # ordered edges
            route_geom = linemerge(list(edges_gdf.geometry))
            total_time_s = float(edges_gdf["travel_time"].sum())
            total_len_m = float(edges_gdf["length"].sum())

            # Great-circle distance between original points (meters)
            gc_m = float(
                ox.distance.great_circle(
                    row["latitude"],
                    row["longitude"],
                    row["lat_next"],
                    row["lon_next"],
                )
            )

            all_routes.append(
                {
                    "person_id": row["person_id"],
                    "step": int(row["step_order"]),
                    "origin_poi": row.get("poi_name", None),
                    "dest_poi": row.get("dest_poi", None),
                    "mode": "directed",
                    "travel_s": total_time_s,
                    "length_m": total_len_m,  # network distance in meters
                    "gc_m": gc_m,  # straight-line distance in meters
                    "geometry": route_geom,
                }
            )

    print("\n✅ Routing complete.")
    print("Zero-length leg details (if any) were written to overlapping.log")
    return all_routes

def multidi_to_best_digraph(G_multi, weight="travel_time"):
    """Keep the minimum-weight edge per (u,v); carry length & geometry onto the simple DiGraph."""
    H = nx.DiGraph()
    H.add_nodes_from((n, {"x": d["x"], "y": d["y"]}) for n, d in G_multi.nodes(data=True))
    for u, v, k, d in G_multi.edges(keys=True, data=True):
        w = d.get(weight)
        if w is None:
            continue
        cur = H.get_edge_data(u, v)
        if (cur is None) or (w < cur.get(weight, float("inf"))):
            H.add_edge(
                u, v,
                travel_time=float(w),
                length=float(d.get("length", 0.0)),
                geometry=d.get("geometry")  # may be None; we’ll synthesize straight segments
            )
    return H


def path_stats_and_geom(H, path):
    """Sum travel_time/length and build a LineString/MultiLineString for a path of nodes."""
    if not path or len(path) < 2:
        return 0.0, 0.0, None
    geoms = []
    tt = 0.0
    L  = 0.0
    for u, v in zip(path[:-1], path[1:]):
        ed = H[u][v]
        tt += float(ed.get("travel_time", 0.0))
        L  += float(ed.get("length", 0.0))
        g = ed.get("geometry")
        if g is None:
            g = LineString([(H.nodes[u]["x"], H.nodes[u]["y"]), (H.nodes[v]["x"], H.nodes[v]["y"])])
        geoms.append(g)
    geom = linemerge(geoms) if len(geoms) > 1 else geoms[0]
    return tt, L, geom


def _reconstruct_path_from_predecessors(pred_map, src, dst):
    """Rebuild node path using predecessor dict (dst -> ... -> src). Returns list or None."""
    if dst not in pred_map or pred_map[dst] == -1:
        return None
    path = [dst]
    cur = dst
    # walk back until src
    while cur != src:
        p = pred_map.get(cur, -1)
        if p == -1 or p is None:
            return None
        path.append(p)
        cur = p
    path.reverse()
    return path
def route_legs_gpu(G_multi, pairs_df):
    """
    GPU-accelerated routing using cuGraph's SSSP:
      - Convert to best-edge DiGraph (H)
      - Build cuGraph DiGraph from H
      - For each origin, run cugraph.sssp once and rebuild paths for its destinations
    """
    try:
        import cudf
        import cugraph
    except Exception as e:
        raise RuntimeError(f"cuGraph not available: {e}")

    print("Routing on GPU with cuGraph (SSSP per origin)...")

    # 1) Collapse to simple DiGraph for path stats & geometry on CPU side
    H = multidi_to_best_digraph(G_multi, weight="travel_time")

    # 2) Build cuGraph DiGraph (renumber=True for performance; we’ll unrenumber results)
    rows = []
    for u, v, d in H.edges(data=True):
        rows.append((np.int64(u), np.int64(v), float(d.get("travel_time", 0.0))))
    gdf = cudf.DataFrame(rows, columns=["src", "dst", "weight"])

    G_cu = cugraph.Graph(directed=True)
    G_cu.from_cudf_edgelist(
        gdf,
        source="src",
        destination="dst",
        edge_attr="weight",
        renumber=True,
    )

    # helper to map an external vertex id to internal (if renumbered)
    def _to_internal(series_like):
        s = cudf.Series(series_like, dtype="int64")
        try:
            return G_cu.lookup_internal_vertex_id(s)
        except Exception:
            # if graph is not renumbered, return as-is
            return s

    # helper to unrenumber 'vertex' / 'predecessor' columns in-place when present
    def _unrenumber(df, cols):
        for col in cols:
            if col in df.columns:
                try:
                    df[col] = G_cu.lookup_external_vertex_id(df[col])
                except Exception:
                    pass
        return df

    all_routes = []

    # group by origin → single SSSP per origin on GPU
    for o_node, group in tqdm(pairs_df.groupby("snap_node"), total=pairs_df["snap_node"].nunique()):
        # 3) Run SSSP from this origin
        src_int = int(_to_internal([int(o_node)]).iloc[0])
        res = cugraph.sssp(G_cu, source=src_int)

        # 4) Unrenumber to external ids (vertex & predecessor)
        res = _unrenumber(res, ["vertex", "predecessor"])

        # move to pandas for quick lookups
        pdf = res.to_pandas()  # columns: vertex, distance, predecessor
        # build a predecessor map for path reconstruction
        pred_map = dict(zip(pdf["vertex"].to_list(), pdf["predecessor"].to_list()))
        dist_map = dict(zip(pdf["vertex"].to_list(), pdf["distance"].to_list()))

        # 5) Emit results for each leg in this origin group
        for _, row in group.iterrows():
            d_node = int(row["snap_node_next"])
            if d_node not in dist_map or np.isinf(dist_map[d_node]):
                continue  # unreachable

            path = _reconstruct_path_from_predecessors(pred_map, int(o_node), d_node)
            if not path or len(path) < 2:
                continue

            tt_s, len_m, geom = path_stats_and_geom(H, path)

            # straight-line distance between original points
            gc_m = float(ox.distance.great_circle(
                row["latitude"], row["longitude"], row["lat_next"], row["lon_next"]
            ))

            all_routes.append({
                "person_id": row["person_id"],
                "step": int(row["step_order"]),
                "origin_poi": row.get("poi_name", None),
                "dest_poi": row.get("dest_poi", None),
                "mode": "directed_gpu",
                "travel_s": tt_s,
                "length_m": len_m,
                "gc_m": gc_m,
                "geometry": geom,
            })

    print("\n✅ GPU routing complete.")
    print("Zero-length leg details (if any) were written to overlapping.log")
    return all_routes

# -------------------- Save -----------------------
def save_routes(all_routes: List[Dict[str, Any]], outfile: str) -> None:
    print("\nStep 5: Converting routes to GeoDataFrame and saving...")
    if not all_routes:
        print("   - No routes were generated, so no file was saved.")
        return
    routes_gdf = gpd.GeoDataFrame(all_routes, crs="EPSG:4326")
    routes_gdf.to_parquet(outfile)
    print(f"Saved to {outfile}")
    print(routes_gdf.head())


# -------------------- Main -----------------------
def main():
    setup_logging()

    # A) AOI (optional downstream use)
    if os.path.exists(AOI_GEOJSON):
        aoi_gs = load_aoi(AOI_GEOJSON)
        aoi_ll = aoi_gs.geometry.iloc[0]  # Available if you need it later
        _ = aoi_ll  # avoid linter unused warning

    # B) Graph
    G = load_graph(GRAPHML_PATH, GPKG_PATH)

    # C) Prep graph
    G, comp_id = prep_graph(G)

    # D) Load trajectories
    traj_df = load_trajectories(TRAJ_CSV)

    # E) Snap to nodes
    traj_df = snap_points_to_nodes(G, traj_df)

    # F) Build pairs (and log zero-length)
    pairs_df = build_pairs_df(G, comp_id, traj_df)

    # G) Routing
    all_routes = route_legs_gpu(G, pairs_df)

    # H) Save
    save_routes(all_routes, ROUTES_OUT)


if __name__ == "__main__":
    main()

# %% [code] In[13]
routes_gdf = gpd.read_parquet("hurricane_ian_routes.parquet")

# %% [code] In[14]
# Ensure step is numeric (just in case), then sort and reset index
routes_gdf['step'] = pd.to_numeric(routes_gdf['step'], errors='coerce')

routes_gdf = routes_gdf.sort_values(
    ['person_id', 'step'],            # sort keys
    ascending=[True, True],           # both ascending
    kind='mergesort'                  # stable sort (keeps equal-step order)
).reset_index(drop=True)
routes_gdf

# %% [code] In[15]
import os, random
import folium

# Ensure output dir exists
output_dir = locals().get("output_dir", "outputs")
os.makedirs(output_dir, exist_ok=True)

if 'routes_gdf' in locals() and isinstance(routes_gdf, gpd.GeoDataFrame) and not routes_gdf.empty:
    print("\nVisualizing routes for a few random individuals...")

    # Make sure geometries are WGS84 for Folium
    routes_wgs = routes_gdf.to_crs(4326) if (routes_gdf.crs and routes_gdf.crs.to_epsg() != 4326) else routes_gdf

    # Map center: centroid of all routes, fallback to trajectories mean
    if routes_wgs.geometry.notnull().any():
        ctr = routes_wgs.unary_union.centroid
        map_center = [ctr.y, ctr.x]
    else:
        map_center = [
            float(trajectories_df['latitude'].mean()),
            float(trajectories_df['longitude'].mean())
        ]

    m_routes = folium.Map(location=map_center, zoom_start=9, tiles="CartoDB Positron")

    # Pick up to 3 people
    unique_ids = routes_wgs['person_id'].dropna().unique().tolist()
    sample_ids = random.sample(unique_ids, min(15, len(unique_ids))) if unique_ids else []

    colors = ['#FF5733', '#337BFF', '#33CC77', '#F333FF', '#FFC300']

    # Track bounds to fit at the end
    overall_bounds = None

    for i, person_id in enumerate(sample_ids):
        color = colors[i % len(colors)]

        # Layer for this person; show only the first one by default
        fg = folium.FeatureGroup(name=f"Person {person_id}", show=(i == 0))

        # Person's routes (lines)
        person_routes = routes_wgs[routes_wgs['person_id'] == person_id]
        if not person_routes.empty:
            folium.GeoJson(
                data=person_routes.__geo_interface__,
                name=f"Routes {person_id}",
                style_function=lambda feature, color=color: {
                    "color": color, "weight": 3, "opacity": 0.9
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=[c for c in ["person_id", "step", "length_m", "travel_s", "gc_m"] if c in person_routes.columns],
                    aliases=["Person", "Step", "Network m", "Travel s", "GC m"],
                    sticky=True
                )
            ).add_to(fg)

            # expand map bounds
            b = person_routes.total_bounds  # [minx, miny, maxx, maxy]
            if overall_bounds is None:
                overall_bounds = list(b)
            else:
                overall_bounds = [
                    min(overall_bounds[0], b[0]),
                    min(overall_bounds[1], b[1]),
                    max(overall_bounds[2], b[2]),
                    max(overall_bounds[3], b[3]),
                ]

        # Person's POIs (dots) + step number labels
        person_pois = trajectories_df[trajectories_df['person_id'] == person_id]
        for _, poi in person_pois.iterrows():
            lat = float(poi['latitude'])
            lon = float(poi['longitude'])
            step = int(poi['step_order']) if pd.notna(poi['step_order']) else None

            # Dot
            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=1.0,
                popup=folium.Popup(
                    f"<b>Person {person_id}</b><br>"
                    f"Step {step if step is not None else '?'}<br>"
                    f"{poi.get('poi_name', '')}",
                    max_width=250
                )
            ).add_to(fg)

            # Step number label (small text next to the dot)
            if step is not None:
                folium.Marker(
                    location=[lat, lon],
                    icon=folium.DivIcon(
                        html=f"""
                        <div style="
                            font-size:10px;
                            font-weight:bold;
                            color:{color};
                            text-shadow:-1px -1px 0 #fff, 1px -1px 0 #fff,
                                        -1px 1px 0 #fff, 1px 1px 0 #fff;
                            transform: translate(8px, -8px);
                        ">{step}</div>"""
                    )
                ).add_to(fg)

        # add this person's layer to the map
        fg.add_to(m_routes)

    # Fit map to all displayed features
    if overall_bounds is not None:
        m_routes.fit_bounds(
            [[overall_bounds[1], overall_bounds[0]], [overall_bounds[3], overall_bounds[2]]]
        )

    folium.LayerControl(collapsed=False).add_to(m_routes)

    # Save and display
    routes_map_path = os.path.join(output_dir, "ian_routes_visualization.html")
    m_routes.save(routes_map_path)
    print(f"🗺️  Route visualization map saved to '{routes_map_path}'")

    try:
        display(m_routes)
    except Exception:
        pass
else:
    print("\nNo routes to visualize.")

# %% [code] In[16]

# %% [code] In[17]
routes_gdf

# %% [code] In[18]
# Randomized hourly-averaged locations with night/home rules
# ----------------------------------------------------------
import re, math, random
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely.ops import substring as _substring
from scipy.stats import beta as _beta

# ---- small compatibility wrapper for shapely substring (normalized) ----
def substring(line: LineString, start_norm: float, end_norm: float) -> LineString:
    try:
        return _substring(line, start_norm, end_norm, normalized=True)
    except Exception:
        # Minimal fallback
        start_norm = max(0.0, min(1.0, start_norm))
        end_norm   = max(0.0, min(1.0, end_norm))
        if end_norm < start_norm: start_norm, end_norm = end_norm, start_norm
        if line.length == 0: return LineString([line.coords[0], line.coords[0]])
        a = line.interpolate(start_norm, normalized=True)
        b = line.interpolate(end_norm,   normalized=True)
        return LineString([a, b])

# ----------------------- Policy knobs (edit as needed) -----------------------
HOURS_TOTAL        = 143
CURFEW_START_H     = 22           # 22:00 local night start
CURFEW_END_H       = 5            # 05:00 "no departure before" (default)
EVAC_DAYS          = {2, 3}       # days (1-indexed from the 1AM start) where early departures allowed
MIN_FIRST_STAY_H   = 4            # at t=1:00, ensure home stay >= 4h (i.e., not leaving before 5am)
RNG_SEED           = 7            # set None for fully random

# dwell time samplers (seconds) ----------------------------------------------
HOME_RE            = re.compile(r'home|residential|apartment|house', re.I)
GROCERY_RE         = re.compile(r'grocery|supermarket', re.I)
RESTAURANT_RE      = re.compile(r'restaurant|eating place|cafe|diner', re.I)

def dwell_sampler(poi_text: str, hour_of_day: float, is_home: bool) -> float:
    """
    Returns a randomized dwell duration (seconds) for a stop just before a leg.
    Uses time-of-day (for home) and POI text.
    """
    # Home logic: at night stay longer; during day keep short if home
    if is_home or (isinstance(poi_text, str) and HOME_RE.search(poi_text or "")):
        if (hour_of_day >= CURFEW_START_H) or (hour_of_day < CURFEW_END_H):
            # night home: ~7-9h
            return float(np.random.uniform(7*3600, 9*3600))
        else:
            # daytime at home: 0.5–1.5h
            return float(np.random.uniform(0.5*3600, 1.5*3600))

    if isinstance(poi_text, str) and GROCERY_RE.search(poi_text):
        # grocery 15–45 min
        return float(np.random.uniform(15*60, 45*60))

    if isinstance(poi_text, str) and RESTAURANT_RE.search(poi_text):
        # restaurant 45–90 min
        return float(np.random.uniform(45*60, 90*60))

    # generic: lognormal-ish 10–60 min
    return float(np.random.uniform(10*60, 60*60))

# travel time randomizer & within-leg speed profile ---------------------------
def travel_time_randomized(travel_s: float) -> float:
    """
    Multiply nominal travel time by a bounded random factor to simulate traffic.
    """
    # 0.8x–1.35x (mild variability)
    f = float(np.random.uniform(0.8, 1.35))
    return max(60.0, travel_s * f)

def make_warp():  # returns a monotone time->distance fraction mapping
    """
    Random monotone warping using a Beta CDF to create uneven speeds along the leg.
    u in [0,1] (time fraction) -> d in [0,1] (distance fraction).
    """
    a = float(np.random.uniform(0.8, 2.2))
    b = float(np.random.uniform(0.8, 2.2))
    dist = _beta(a=a, b=b)
    def warp(u):
        u = max(0.0, min(1.0, float(u)))
        return float(dist.cdf(u))  # already in [0,1]
    return warp

# ----------------------- core helper: hourly aggregation ----------------------
def _time_to_day_hour(t_s: float):
    """Day index (1-based) and hour-of-day [0..24), given t=0 at 01:00 on day 1."""
    hour_since_start = t_s / 3600.0
    day = int(hour_since_start // 24) + 1
    hod = (1.0 + hour_since_start) % 24.0
    return day, hod

def _ensure_wgs84(gdf: gpd.GeoDataFrame):
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        return gdf.to_crs(4326)
    return gdf

def _pick_home_point(dfp: gpd.GeoDataFrame) -> Point:
    # Prefer first origin that looks like home; else first line start
    for _, r in dfp.iterrows():
        if isinstance(r.get("origin_poi", None), str) and HOME_RE.search(r["origin_poi"]):
            line = r.geometry
            if isinstance(line, LineString):
                return Point(line.coords[0])
    # fallback
    first_line = dfp.iloc[0].geometry
    return Point(first_line.coords[0])

def _build_intervals_with_policy(dfp: gpd.GeoDataFrame) -> list[tuple[str, object, float]]:
    """
    Build a list of intervals for one person with:
      ("stay", Point, seconds) or ("move", LineString, seconds, warp_fn)
    Policy: start at home >= MIN_FIRST_STAY_H; no departures before 05:00 except EVAC_DAYS.
    Insert nightly home stays where needed.
    """
    intervals = []
    t = 0.0  # absolute seconds since start (01:00 day 1)
    home_pt = _pick_home_point(dfp)

    # 1) initial stay at home (>= 5am)
    intervals.append(("stay", home_pt, MIN_FIRST_STAY_H * 3600.0))
    t += MIN_FIRST_STAY_H * 3600.0

    # 2) iterate legs
    for _, row in dfp.iterrows():
        line: LineString = row.geometry
        if not isinstance(line, LineString):
            continue

        # Night rule: if current local time is inside curfew and it's NOT an evac day,
        # add stay-at-home until 05:00.
        day, hod = _time_to_day_hour(t)
        if (day not in EVAC_DAYS) and (hod < CURFEW_END_H):
            add_s = (CURFEW_END_H - hod) * 3600.0
            intervals.append(("stay", home_pt, add_s))
            t += add_s
            day, hod = _time_to_day_hour(t)

        if (day not in EVAC_DAYS) and (hod >= CURFEW_START_H):
            # already late night, push to next day 05:00
            add_s = ((24 - hod) + CURFEW_END_H) * 3600.0
            intervals.append(("stay", home_pt, add_s))
            t += add_s
            day, hod = _time_to_day_hour(t)

        # 2a) dwell at origin (random, POI-aware)
        origin_txt = row.get("origin_poi", "")
        is_home = bool(isinstance(origin_txt, str) and HOME_RE.search(origin_txt))
        dwell_s = dwell_sampler(origin_txt, hod, is_home)
        intervals.append(("stay", Point(line.coords[0]), dwell_s))
        t += dwell_s
        day, hod = _time_to_day_hour(t)

        # 2b) travel with randomized duration and uneven speed
        travel_s = float(row.get("travel_s", 0.0))
        if travel_s > 0:
            dur = travel_time_randomized(travel_s)
            warp = make_warp()
            intervals.append(("move", line, dur, warp))
            t += dur
            day, hod = _time_to_day_hour(t)

    # 3) end with a reasonable dwell at the final destination (home if it is)
    last_line = dfp.iloc[-1].geometry
    last_dest_txt = dfp.iloc[-1].get("dest_poi", "")
    end_pt = Point(last_line.coords[-1])
    _, hod = _time_to_day_hour(t)
    is_home_end = bool(isinstance(last_dest_txt, str) and HOME_RE.search(last_dest_txt))
    intervals.append(("stay", end_pt, dwell_sampler(last_dest_txt, hod, is_home_end)))

    return intervals

def hourly_avg_locations_randomized(
    trips_gdf: gpd.GeoDataFrame,
    hours_total: int = HOURS_TOTAL,
    person_col: str = "person_id",
    step_col: str = "step",
    origin_poi_col: str = "origin_poi",
    dest_poi_col: str = "dest_poi",
    travel_s_col: str = "travel_s",
    geometry_col_guess: str = "geom",
    seed: int | None = RNG_SEED,
    sample_every_s: int = 300,  # 5-min time sampling inside each hour for moving legs
):
    """
    Returns (long_gdf, wide_df)
      long_gdf: [trajectory_id, hour, geometry(Point EPSG:4326)]
      wide_df : [trajectory_id, h001..h{hours_total}] with shapely Points
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    gdf = trips_gdf.copy()
    # normalize geometry column name
    if geometry_col_guess in gdf.columns and "geometry" not in gdf:
        gdf = gdf.set_geometry(geometry_col_guess)
    if gdf.geometry.name != "geometry":
        gdf = gdf.set_geometry(gdf.geometry.name)

    gdf = _ensure_wgs84(gdf)
    gdf = gdf.rename(columns={
        origin_poi_col: "origin_poi",
        dest_poi_col:   "dest_poi",
        travel_s_col:   "travel_s",
    })
    gdf = gdf.sort_values([person_col, step_col]).reset_index(drop=True)

    hour_len = 3600.0
    hour_bounds = [(i*hour_len, (i+1)*hour_len) for i in range(hours_total)]
    max_T = hours_total * hour_len

    out = []
    for pid, dfp in gdf.groupby(person_col, sort=False):
        dfp = dfp.reset_index(drop=True)
        intervals = _build_intervals_with_policy(dfp)

        # build cumulative times, clipped to max_T
        ends = []
        acc = 0.0
        packed = []
        for it in intervals:
            if it[0] == "stay":
                _, pt, dur = it
                dur = float(dur)
                if dur <= 0: continue
                if acc >= max_T: break
                if acc + dur > max_T: dur = max_T - acc
                packed.append(("stay", pt, dur))
                acc += dur
                ends.append(acc)
            else:
                _, line, dur, warp = it
                dur = float(dur)
                if dur <= 0: continue
                if acc >= max_T: break
                if acc + dur > max_T: dur = max_T - acc
                packed.append(("move", line, dur, warp))
                acc += dur
                ends.append(acc)
        if not packed:
            continue

        # per-hour accumulation
        j = 0  # current interval index
        prev_end = 0.0
        for h, (hs, he) in enumerate(hour_bounds, start=1):
            wx = wy = wsum = 0.0

            # skip finished hours
            while j < len(packed) and ends[j] <= hs:
                prev_end = ends[j]
                j += 1

            k = j
            while k < len(packed):
                seg = packed[k]
                seg_start = (ends[k-1] if k > 0 else 0.0)
                seg_end   = ends[k]
                ov_start  = max(hs, seg_start)
                ov_end    = min(he, seg_end)
                if ov_end <= ov_start:
                    if seg_start >= he:
                        break
                    k += 1
                    continue

                ov = ov_end - ov_start
                if seg[0] == "stay":
                    pt: Point = seg[1]
                    wx += pt.x * ov
                    wy += pt.y * ov
                    wsum += ov
                else:
                    line: LineString = seg[1]
                    dur              = seg[2]
                    warp_fn          = seg[3]

                    # time sampling inside the overlapped window
                    n = max(3, int(ov / sample_every_s))
                    dt = ov / n
                    for s in range(n):
                        t_mid = ov_start + (s + 0.5) * dt
                        tau   = (t_mid - seg_start) / dur  # time fraction in [0,1]
                        dfrac = warp_fn(tau)               # distance fraction in [0,1]
                        p     = line.interpolate(dfrac, normalized=True)
                        wx += p.x * dt
                        wy += p.y * dt
                        wsum += dt

                if seg_end >= he:
                    break
                k += 1

            if wsum > 0:
                lon, lat = wx/wsum, wy/wsum
                out.append((pid, h, Point(lon, lat)))
            else:
                # if nothing overlapped (should be rare), carry last position for this pid
                if len(out) and out[-1][0] == pid:
                    out.append((pid, h, out[-1][2]))
                else:
                    # fallback to that person's home
                    home_pt = _pick_home_point(dfp)
                    out.append((pid, h, home_pt))

    long_gdf = gpd.GeoDataFrame(out, columns=["trajectory_id", "hour", "geometry"],
                                geometry="geometry", crs="EPSG:4326")

    wide_df = (long_gdf
               .assign(location=lambda d: d.geometry)
               .pivot_table(index="trajectory_id", columns="hour", values="location", aggfunc="first")
               .rename(columns=lambda h: f"h{int(h):03d}")
               .reset_index()
               .sort_values("trajectory_id"))

    return long_gdf, wide_df

# %% [code] In[19]
traj_summary_df

# %% [code] In[20]

# %% [code] In[21]
routes_gdf

# %% [code] In[22]
# Your trips table must be a GeoDataFrame with these columns:
# person_id, step, origin_poi, dest_poi, travel_s, and a LineString route in 'geom' (or 'geometry')
# If CRS isn’t EPSG:4326, it will be converted.

long_gdf, wide_df = hourly_avg_locations_randomized(
    routes_gdf,                      # your GeoDataFrame
    hours_total=143,
    person_col="person_id",
    step_col="step",
    origin_poi_col="origin_poi",
    dest_poi_col="dest_poi",
    travel_s_col="travel_s",
    geometry_col_guess="geom",
    seed=7,                   # set None for different randomization each run
    sample_every_s=300        # 5-min resolution for within-hour motion
)

# long_gdf: [trajectory_id, hour, geometry(Point in degrees)]
# wide_df : [trajectory_id, h001..h143] (each cell is a shapely Point)

# %% [code] In[23]
wide_df

# %% [code] In[24]
import re
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString

# --- 1) Attach cluster info to wide_df --------------------------------------
def attach_cluster_to_wide(wide_df: pd.DataFrame, traj_summary_df: pd.DataFrame) -> pd.DataFrame:
    wd = wide_df.copy()

    # Normalize the ID column name on wide_df: prefer person_id
    if "person_id" not in wd.columns:
        if "trajectory_id" in wd.columns:
            wd = wd.rename(columns={"trajectory_id": "person_id"})
        elif "hour" in wd.columns:                      # per your request
            wd = wd.rename(columns={"hour": "person_id"})
        else:
            raise KeyError("wide_df needs an id column: one of ['person_id','trajectory_id','hour'].")

    # Normalize the ID column name on traj_summary_df
    ts = traj_summary_df.copy()
    if "person_id" not in ts.columns and "trajectory_id" in ts.columns:
        ts = ts.rename(columns={"trajectory_id": "person_id"})
    if "traj_cluster" not in ts.columns:
        raise KeyError("traj_summary_df must contain 'traj_cluster'.")

    ts = ts[["person_id", "traj_cluster"]].drop_duplicates("person_id")

    # Left-join cluster onto wide_df
    wd = wd.merge(ts, on="person_id", how="left")

    # Quick sanity check
    missing = wd["traj_cluster"].isna().sum()
    if missing:
        print(f"[warn] {missing} rows in wide_df have no matching traj_cluster.")
    return wd

wide_df_with_cluster = attach_cluster_to_wide(wide_df, traj_summary_df)

# %% [code] In[25]
# --- Layer-per-person map WITH an hourly dot (h001..h143) for each person ---
import os, re, random
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
import folium
from folium.plugins import MarkerCluster

# -------- settings --------
OUTPUT_DIR        = locals().get("output_dir", "outputs")
MAX_PERSON_LAYERS = 80      # set None to include all persons (may be heavy)
SHOW_FIRST_N      = 3       # how many person layers are "on" by default
MAP_TILES         = "CartoDB Positron"
ZOOM_START        = 9
USE_MARKER_CLUSTER = True   # cluster the hour dots per person for performance
DOT_RADIUS        = 3       # radius of each hourly point (ignored when clustering)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------- helpers --------
def _hour_cols(df):  # h001..h143
    return sorted([c for c in df.columns if re.fullmatch(r"h\d{3}", c)])

def attach_cluster_to_wide(wide_df: pd.DataFrame, traj_summary_df: pd.DataFrame|None=None) -> pd.DataFrame:
    wd = wide_df.copy()
    if "person_id" not in wd.columns:
        if "trajectory_id" in wd.columns:
            wd = wd.rename(columns={"trajectory_id": "person_id"})
        elif "hour" in wd.columns:
            wd = wd.rename(columns={"hour": "person_id"})
        else:
            raise KeyError("wide_df needs an id col: one of ['person_id','trajectory_id','hour'].")
    if traj_summary_df is not None and not traj_summary_df.empty:
        ts = traj_summary_df.copy()
        if "person_id" not in ts.columns and "trajectory_id" in ts.columns:
            ts = ts.rename(columns={"trajectory_id": "person_id"})
        ts = ts[["person_id","traj_cluster"]].drop_duplicates("person_id")
        wd = wd.merge(ts, on="person_id", how="left")
    wd["traj_cluster"] = wd.get("traj_cluster", "unknown").fillna("unknown")
    return wd

def build_lines_from_wide(wd: pd.DataFrame) -> gpd.GeoDataFrame:
    hour_cols = _hour_cols(wd)
    if not hour_cols:
        raise ValueError("No hour columns like h001..h143 in wide_df.")
    lines, ids, clusters = [], [], []
    for _, row in wd.iterrows():
        pts = [row[c] for c in hour_cols]
        coords = [(p.x, p.y) for p in pts if isinstance(p, Point) and np.isfinite(getattr(p, "x", np.nan)) and np.isfinite(getattr(p, "y", np.nan))]
        if len(coords) >= 2:
            lines.append(LineString(coords))
            ids.append(row["person_id"])
            clusters.append(row.get("traj_cluster", "unknown"))
    if not lines:
        raise ValueError("No valid trajectories (need ≥2 points).")
    return gpd.GeoDataFrame({"person_id": ids, "traj_cluster": clusters}, geometry=lines, crs="EPSG:4326")

def hourly_points_for_person(row: pd.Series) -> list[tuple[int, float, float]]:
    """Return [(hour_index, lat, lon), ...] for valid hourly Points in the row."""
    out = []
    for col in _hour_cols(pd.DataFrame([row])):
        p = row[col]
        if isinstance(p, Point) and np.isfinite(getattr(p, "x", np.nan)) and np.isfinite(getattr(p, "y", np.nan)):
            hour_idx = int(col[1:])  # h001 -> 1
            out.append((hour_idx, p.y, p.x))  # (lat, lon)
    return out

def color_for_id(pid):
    palette = [
        "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
        "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"
    ]
    return palette[abs(hash(str(pid))) % len(palette)]

# -------- prepare data --------
if "wide_df_with_cluster" not in locals():
    if "wide_df" in locals():
        wide_df_with_cluster = attach_cluster_to_wide(wide_df, locals().get("traj_summary_df"))
    else:
        raise NameError("Please define wide_df (and optionally traj_summary_df) first.")

lines_gdf = build_lines_from_wide(wide_df_with_cluster)

# sample persons (optional)
persons = lines_gdf["person_id"].unique().tolist()
if MAX_PERSON_LAYERS is not None and len(persons) > MAX_PERSON_LAYERS:
    random.seed(0)
    persons = random.sample(persons, MAX_PERSON_LAYERS)

lines_plot = lines_gdf[lines_gdf["person_id"].isin(persons)].copy()

# map center/bounds
ctr = lines_plot.unary_union.centroid
center = [float(ctr.y), float(ctr.x)]
bounds = lines_plot.total_bounds  # [minx, miny, maxx, maxy]

# -------- build map (one FeatureGroup per person + hourly dots) --------
m = folium.Map(location=center, zoom_start=ZOOM_START, tiles=MAP_TILES)

# quick lookup of each person's wide row (for hourly points)
row_by_pid = {int(r["person_id"]) if str(r["person_id"]).isdigit() else r["person_id"]: r
              for _, r in wide_df_with_cluster.set_index("person_id").reset_index().iterrows()}

for i, pid in enumerate(persons):
    sub = lines_plot[lines_plot["person_id"] == pid]
    if sub.empty:
        continue

    line: LineString = sub.iloc[0].geometry
    cl = str(sub.iloc[0].get("traj_cluster", "unknown"))
    color = color_for_id(pid)

    fg = folium.FeatureGroup(name=f"Person {pid} (Cluster {cl})", show=(i < SHOW_FIRST_N))

    # polyline
    coords_latlon = [(lat, lon) for lon, lat in line.coords]
    folium.PolyLine(
        locations=coords_latlon,
        weight=3, opacity=0.9, color=color,
        tooltip=folium.Tooltip(f"Person {pid} | Cluster {cl}")
    ).add_to(fg)

    # start / end markers
    start_latlon = (line.coords[0][1],  line.coords[0][0])
    end_latlon   = (line.coords[-1][1], line.coords[-1][0])
    folium.CircleMarker(location=start_latlon, radius=4, color=color, fill=True,
                        fill_opacity=1.0, tooltip=f"Start • {pid}").add_to(fg)
    folium.CircleMarker(location=end_latlon, radius=4, color=color, fill=True,
                        fill_opacity=0.25, tooltip=f"End • {pid}").add_to(fg)

    # --- hourly dots (one for each available hour point) ---
    row = row_by_pid.get(pid)
    if row is not None:
        hour_pts = hourly_points_for_person(row)

        if USE_MARKER_CLUSTER:
            cluster = MarkerCluster(name=f"Hours • {pid}", show=False)  # toggled when person layer is on
            for h, lat, lon in hour_pts:
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=DOT_RADIUS, color=color, fill=True, fill_opacity=0.9,
                    tooltip=folium.Tooltip(f"Person {pid} • hr {h:03d}<br>({lat:.5f}, {lon:.5f})")
                ).add_to(cluster)
            cluster.add_to(fg)
        else:
            for h, lat, lon in hour_pts:
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=DOT_RADIUS, color=color, fill=True, fill_opacity=0.9,
                    tooltip=folium.Tooltip(f"Person {pid} • hr {h:03d}<br>({lat:.5f}, {lon:.5f})")
                ).add_to(fg)

    fg.add_to(m)

# controls & save
m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
folium.LayerControl(collapsed=False).add_to(m)

out_path = os.path.join(OUTPUT_DIR, "hourly_trajectories_by_person_with_dots.html")
m.save(out_path)
print(f"🗺️  Per-person map with hourly dots saved to: {out_path}")

try:
    display(m)
except Exception:
    pass

# %% [code] In[26]
wide_df_with_cluster.to_parquet("hourlymap.parquet")

# %% [code] In[27]
import re
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# --- 1) pick the hour columns ---
hour_cols = sorted([c for c in wide_df_with_cluster.columns if re.fullmatch(r"h\d{3}", c)])
if not hour_cols:
    raise ValueError("No hour columns (h001..h143) found in wide_df_with_cluster.")

# --- 2) melt -> one row per (person_id, hour) with geometry = Point ---
id_cols = ["person_id"]
if "traj_cluster" in wide_df_with_cluster.columns:
    id_cols.append("traj_cluster")

long_df = (wide_df_with_cluster[id_cols + hour_cols]
           .melt(id_vars=id_cols, value_vars=hour_cols,
                 var_name="hcol", value_name="geometry"))

# keep valid Points only
is_point = long_df["geometry"].apply(lambda g: isinstance(g, Point))
long_df = long_df[is_point].copy()

# --- 3) add fields to match your target schema ---
long_df["traj_id"] = long_df["person_id"]                      # or map to your own ID
long_df["pt_idx"]  = long_df["hcol"].str[1:].astype(int)       # h001 -> 1, ... h143 -> 143
long_df["longitude"] = long_df["geometry"].apply(lambda p: p.x)
long_df["latitude"]  = long_df["geometry"].apply(lambda p: p.y)

# --- 4) build the GeoDataFrame in WGS84 and order columns like the screenshot ---
point_gdf = gpd.GeoDataFrame(
    long_df[["traj_id", "pt_idx", "latitude", "longitude", "geometry"]].reset_index(drop=True),
    geometry="geometry",
    crs="EPSG:4326"
)

# %% [code] In[28]
point_gdf.to_parquet("simulated_traj_points.parquet")

# %% [code] In[29]
blist = wide_df_with_cluster.traj_cluster.to_list()

# %% [code] In[30]
import pickle

# %% [code] In[31]
with open("ground_t.pickle", 'wb') as f:
    pickle.dump(blist,f)

