# Auto-generated from notebook; edit as Python module if needed.
# You can refactor functions into the package (evacmob) and import them in scripts/.

# %% [code] In[1]
import numpy as np
import pandas as pd

# %% [code] In[2]
import geopandas as gpd

# %% [code] In[3]
import os
import math
import logging
import pickle

import numpy as np
import pandas as pd
import geopandas as gpd

from shapely import wkt
from shapely.geometry import Point

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM  # Assuming these might be used later from original imports
from peft import PeftModel

# %% [code] In[4]
df = pd.read_csv("DRIVES_data.csv")

# %% [code] In[5]
df_new = df.iloc[:-1] 

# %% [code] In[6]
df_new.isna().sum(axis=0)

# %% [code] In[7]
len(df_new.participantId.unique().tolist())

# %% [code] In[8]
poi_df = pd.read_csv("../US_POI.csv")

# %% [code] In[9]
poi_df

# %% [code] In[10]
df_new

# %% [code] In[11]
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
from pyproj import Geod

# ---------- 1) Make a GeoDataFrame with start & end points (EPSG:4326) ----------
def make_points_gdf(df):
    df = df.copy()

    # Parse datetimes (assume UTC strings like "2022-07-01 00:00:32+00:00")
    for c in ["tripStartDate", "tripEndDate"]:
        df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")

    # Build point geometries
    df["start_geom"] = [
        Point(lon, lat) if pd.notnull(lon) and pd.notnull(lat) else None
        for lon, lat in zip(df["tripStartLongitude"], df["tripStartLatitude"])
    ]
    df["end_geom"] = [
        Point(lon, lat) if pd.notnull(lon) and pd.notnull(lat) else None
        for lon, lat in zip(df["tripEndLongitude"], df["tripEndLatitude"])
    ]

    # Keep one active geometry (start); end is stored as an extra column
    gdf = gpd.GeoDataFrame(df, geometry="start_geom", crs="EPSG:4326")

    # Clean obvious nulls
    gdf = gdf.dropna(subset=["start_geom", "end_geom", "tripStartDate", "tripEndDate"])

    return gdf



# %% [code] In[12]
gdf_pts = make_points_gdf(df_new)

# %% [code] In[13]
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
from pyproj import Geod

_geod = Geod(ellps="WGS84")

def _geo_dist_m(p1: Point, p2: Point) -> float:
    _, _, dist_m = _geod.inv(p1.x, p1.y, p2.x, p2.y)
    return dist_m

def stitch_trips_to_lines_with_gaps(
    gdf_points: gpd.GeoDataFrame,
    person_col: str = "participantId",
    join_dist_m: float = 100.0,
    join_gap_max: pd.Timedelta = pd.Timedelta("4H"),
    hard_break_gap: pd.Timedelta = pd.Timedelta("8H"),
):
    """
    Same logic as before, but also records the within-segment 'links' (gaps).
    Returns (gdf_lines, gdf_links), both EPSG:4326.
    """
    cols_needed = [person_col, "tripStartDate", "tripEndDate", "start_geom", "end_geom"]
    missing = [c for c in cols_needed if c not in gdf_points.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    seg_rows = []
    link_rows = []

    def _process_person(dfp: pd.DataFrame):
        dfp = dfp.sort_values("tripStartDate").reset_index(drop=True)

        seg_id = 0
        seg_points = []
        seg_trips = []
        seg_start_time = None
        seg_end_time = None
        last_end_point = None
        last_end_time = None

        # per-segment link accumulators
        seg_gaps_s = []
        seg_dists_m = []
        link_idx = 0

        def flush_segment(tag: str = "final"):
            nonlocal seg_id, seg_points, seg_trips, seg_start_time, seg_end_time
            nonlocal seg_gaps_s, seg_dists_m, link_idx

            if len(seg_points) >= 2:
                # summary stats over the links inside this segment
                link_count = len(seg_gaps_s)
                if link_count > 0:
                    max_gap_h = max(seg_gaps_s)/3600.0
                    mean_gap_h = (sum(seg_gaps_s)/link_count)/3600.0
                    total_gap_h = sum(seg_gaps_s)/3600.0
                else:
                    max_gap_h = mean_gap_h = total_gap_h = 0.0

                seg_rows.append({
                    person_col: dfp.iloc[0][person_col],
                    "seg_id": seg_id,
                    "num_trips": len(seg_trips),
                    "link_count": link_count,
                    "start_time": seg_start_time,
                    "end_time": seg_end_time,
                    "max_gap_h": max_gap_h,
                    "mean_gap_h": mean_gap_h,
                    "total_gap_h": total_gap_h,
                    "segment_type": tag,
                    "geometry": LineString(seg_points),
                })

            # reset for next segment
            seg_id += 1
            seg_points = []
            seg_trips = []
            seg_gaps_s = []
            seg_dists_m = []
            link_idx = 0

        for i, row in dfp.iterrows():
            s_pt, e_pt = row["start_geom"], row["end_geom"]
            s_t, e_t = row["tripStartDate"], row["tripEndDate"]

            if last_end_point is None:
                # start first segment
                seg_points = [s_pt, e_pt]
                seg_trips = [i]
                seg_start_time, seg_end_time = s_t, e_t
                last_end_point, last_end_time = e_pt, e_t
                continue

            gap = s_t - last_end_time
            dist = _geo_dist_m(last_end_point, s_pt)

            if pd.notnull(gap) and gap <= join_gap_max and dist <= join_dist_m:
                # continue this segment
                # record the link (prev end -> this start)
                gap_s = gap.total_seconds()
                link_rows.append({
                    person_col: dfp.iloc[0][person_col],
                    "seg_id": seg_id,
                    "link_idx": link_idx,
                    "stop_time": last_end_time,
                    "next_start_time": s_t,
                    "gap_s": gap_s,
                    "gap_h": gap_s/3600.0,
                    "dist_m": dist,
                    "geometry": LineString([last_end_point, s_pt]),
                })
                link_idx += 1
                seg_gaps_s.append(gap_s)
                seg_dists_m.append(dist)

                # extend segment with the new trip's end
                seg_points.append(e_pt)
                seg_trips.append(i)
                seg_end_time = e_t
                last_end_point, last_end_time = e_pt, e_t
            else:
                # break; classify soft vs hard
                tag = "soft_break" if (gap <= hard_break_gap and dist <= join_dist_m) else "hard_break"
                flush_segment(tag=tag)

                # start new segment with this trip
                seg_points = [s_pt, e_pt]
                seg_trips = [i]
                seg_start_time, seg_end_time = s_t, e_t
                last_end_point, last_end_time = e_pt, e_t

        # flush trailing segment
        flush_segment(tag="final")

    for _, dfp in gdf_points.groupby(person_col, sort=False):
        _process_person(dfp)

    gdf_lines = gpd.GeoDataFrame(seg_rows, geometry="geometry", crs="EPSG:4326")
    gdf_links = gpd.GeoDataFrame(link_rows, geometry="geometry", crs="EPSG:4326")
    return gdf_lines, gdf_links

# %% [code] In[14]
gdf_lines, gdf_links = stitch_trips_to_lines_with_gaps(
    gdf_pts,
    person_col="participantId",
    join_dist_m=100.0,
    join_gap_max=pd.Timedelta("4H"),
    hard_break_gap=pd.Timedelta("8H"),
)

# %% [code] In[15]
gdf_links

# %% [code] In[16]
gdf_pts

# %% [code] In[17]
poi_df

# %% [code] In[18]
import numpy as np
import pandas as pd
from shapely.geometry import LineString

# ---------- helpers ----------
def _resolve_col(df, candidates):
    """Find a column (case-insensitive) from a list of candidate names."""
    cols = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in cols:
            return cols[name.lower()]
    raise KeyError(f"None of {candidates} found in columns: {list(df.columns)}")

def _haversine_m(lat1, lon1, lat2, lon2):
    """
    Vectorized haversine distance in meters.
    lat/lon in degrees. lat1/lon1 can be scalars; lat2/lon2 can be numpy arrays.
    """
    R = 6371000.0  # meters
    lat1 = np.deg2rad(lat1); lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2); lon2 = np.deg2rad(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2.0*np.arcsin(np.sqrt(a))
    return R * c

def nearest_pois_for_links(gdf_links, poi_df, k=10, n_links=15,
                           person_col="participantId"):
    # Resolve POI columns (case-insensitive)
    lat_col  = _resolve_col(poi_df, ["LATITUDE", "latitude", "lat"])
    lon_col  = _resolve_col(poi_df, ["LONGITUDE", "longitude", "lon", "lng"])
    name_col = _resolve_col(poi_df, ["LOCATION_NAME", "location_name", "name"])
    top_col  = _resolve_col(poi_df, ["TOP_CATEGORY", "top_category"])
    sub_col  = _resolve_col(poi_df, ["SUB_CATEGORY", "sub_category"])
    key_col  = _resolve_col(poi_df, ["PLACEKEY", "placekey"])

    # Filter POIs with valid coords
    pois = poi_df.dropna(subset=[lat_col, lon_col]).copy()
    poi_lat = pois[lat_col].to_numpy(dtype=float)
    poi_lon = pois[lon_col].to_numpy(dtype=float)

    out_rows = []

    # Work on the first n_links
    links = gdf_links.iloc[:n_links].reset_index(drop=True)

    for i, row in links.iterrows():
        # Pull meta
        pid     = row.get(person_col, None)
        seg_id  = row.get("seg_id", None)
        link_id = row.get("link_idx", i)

        # Get start/end points from the LineString geometry
        geom = row.geometry
        if not isinstance(geom, LineString) or geom.is_empty:
            continue
        (start_lon, start_lat) = geom.coords[0]
        (end_lon,   end_lat)   = geom.coords[-1]

        # For each endpoint (start, end), compute distances to all POIs and take top-k
        for endpoint, (qlat, qlon) in [("start", (start_lat, start_lon)),
                                       ("end",   (end_lat,   end_lon))]:
            dists = _haversine_m(qlat, qlon, poi_lat, poi_lon)

            # Select k smallest efficiently, then sort
            k_eff = min(k, dists.size)
            idx_k = np.argpartition(dists, k_eff-1)[:k_eff]
            idx_sorted = idx_k[np.argsort(dists[idx_k])]

            # Record rows
            for rank, j in enumerate(idx_sorted, start=1):
                out_rows.append({
                    person_col: pid,
                    "seg_id": seg_id,
                    "link_idx": link_id,
                    "endpoint": endpoint,     # "start" or "end"
                    "rank": rank,             # 1..k
                    "poi_placekey": pois.iloc[j][key_col],
                    "poi_name": pois.iloc[j][name_col],
                    "top_category": pois.iloc[j][top_col],
                    "sub_category": pois.iloc[j][sub_col],
                    "poi_lat": float(pois.iloc[j][lat_col]),
                    "poi_lon": float(pois.iloc[j][lon_col]),
                    "distance_m": float(dists[j]),
                    # query point for reference
                    "q_lat": float(qlat),
                    "q_lon": float(qlon),
                })

    result = pd.DataFrame(out_rows).sort_values(
        ["participantId", "seg_id", "link_idx", "endpoint", "rank"],
        kind="stable"
    )
    return result

# ---------- run it ----------
# Make sure gdf_links is EPSG:4326 and poi_df has valid coords
nearest_15x10 = nearest_pois_for_links(gdf_links, poi_df, k=10, n_links=15,
                                       person_col="participantId")

# Peek
nearest_15x10.head(50)

# %% [code] In[19]
import geopandas as gpd
import pandas as pd

# --- 0) Assumptions ---
# gdf_pts has EPSG:4326 and two point columns: start_geom, end_geom
# poi_df has columns LATITUDE/LONGITUDE (or similar)

# --- 1) Build a single GeoSeries of all points (start + end) ---
all_pts = pd.concat([gdf_pts["start_geom"], gdf_pts["end_geom"]], ignore_index=True).dropna()
all_pts = gpd.GeoSeries(all_pts, crs="EPSG:4326")

# (optional but recommended) project to a planar CRS before computing hull,
# then convert back to WGS84 to avoid geodesic distortions.
# For US-wide data, EPSG:5070 (NAD83 / Conus Albers) works well.
all_pts_proj = all_pts.to_crs(5070)
hull_proj = all_pts_proj.unary_union.convex_hull
hull_4326 = gpd.GeoSeries([hull_proj], crs=5070).to_crs(4326).iloc[0]

# --- 2) Turn POIs into a GeoDataFrame ---
def _resolve_col(df, candidates):
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    raise KeyError(f"Need one of {candidates}")

lat_col = _resolve_col(poi_df, ["LATITUDE","latitude","lat"])
lon_col = _resolve_col(poi_df, ["LONGITUDE","longitude","lon","lng"])

poi_gdf = gpd.GeoDataFrame(
    poi_df.copy(),
    geometry=gpd.points_from_xy(poi_df[lon_col], poi_df[lat_col]),
    crs="EPSG:4326"
)

# --- 3) (Fast pre-filter) Clip by hull bounding box, then precise 'within' test ---
minx, miny, maxx, maxy = hull_4326.bounds
candidate_pois = poi_gdf.cx[minx:maxx, miny:maxy]   # uses spatial index when available

# precise point-in-polygon
pois_in_hull = candidate_pois[candidate_pois.within(hull_4326)].copy()

print(f"POIs total: {len(poi_gdf):,}")
print(f"POIs in hull: {len(pois_in_hull):,}")

# If you want the hull as a one-row GeoDataFrame for plotting / saving:
hull_gdf = gpd.GeoDataFrame({"name":["trip_convex_hull"]}, geometry=[hull_4326], crs="EPSG:4326")

# %% [code] In[20]
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
# 4) Move to GPU/CPU and set eval mode:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 1) Where you saved your LoRA adapters after training:
CHECKPOINT = "/storage1/fs1/nlin/Active/sizhe/FO_DATA/checkpoint-dir/checkpoint-551"

# 2) Load the tokenizer from that folder (it contains tokenizer.json + vocab.txt)
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

base_name = "google/gemma-3-1b-it"     # <— exactly what you passed in your training script
base = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-1b-it",
    attn_implementation="eager"
).to(device)

# 3) Now graft the adapter weights:
llm_model = PeftModel.from_pretrained(base, CHECKPOINT)


llm_model.to(device).eval()


# %% [code] In[21]

# %% [code] In[22]
def embed_texts(texts: list[str]) -> torch.Tensor:
    """
    texts: a Python list of strings (each may contain literal "[sep]").
    example: ["[Museums, Historical Sites,...<sep>Museums, Historical Sites<sep>Natural History museum]",...]
    In the format of [<category><sep><subcategory><sep><name>]
    Returns: a GPU tensor of shape (len(texts), hidden_size) containing the
             final‐token embedding for each string.
    """
    # 2.1) Clean & force everything to str, replacing None/NaN with ""
    clean_texts=[]
    for t in texts:
        if t is None:
            clean_texts.append("")
        else:
            clean_texts.append(str(t))
    sep = tokenizer.sep_token  # e.g. "[SEP]" for BERT‐style; whatever your tokenizer.sep_token is
    clean_texts = [t.replace("[sep]", sep) for t in clean_texts]
    
    # 2.3) Batch‐tokenize all strings onto GPU
    enc = tokenizer(
        clean_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
        return_token_type_ids=False,
    ).to(device)
    #print("Batch input_ids are on:", enc.input_ids.device)
    # Now enc.input_ids and enc.attention_mask live on GPU.

    # 2.4) Forward‐pass (no gradients) to get hidden states
    with torch.no_grad():
        outputs = llm_model.base_model(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            output_hidden_states=True
        )
        # `outputs.hidden_states` is a tuple of length (num_layers+1);
        # each element has shape (batch_size, seq_len, hidden_size).
        last_hidden = outputs.hidden_states[-1]  # (batch_size, seq_len, hidden_size) on GPU

    # 2.5) For each sequence, pick out the final non‐pad token embedding
    seq_lens = enc.attention_mask.sum(dim=1) - 1  # (batch_size,) on GPU, index of last non-pad
    batch_size, hidden_size = last_hidden.size(0), last_hidden.size(2)

    # Gather the embedding at position (i, seq_lens[i], :)
    final_embs = last_hidden[torch.arange(batch_size), seq_lens, :]  # (batch_size, hidden_size) on GPU
    #print((final_embs))
    return final_embs

# %% [code] In[23]
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
null_emb   = embed_texts(["<null_val>[sep]<null_val>"]) 
# null_emb is already a tensor
print(type(null_emb), null_emb.shape, null_emb.device)

# add the batch‐dim, cast & move to DEVICE
null_tensor = null_emb.float().unsqueeze(0).to(DEVICE)
print(null_tensor, null_tensor.device)

# %% [code] In[24]

ckpt = torch.load("bottleneck_mlp_newdata.pth",map_location = "cuda",weights_only=False)

raw_classes = ckpt.get("classes", None)
# Determine number of classes
n_old = len(raw_classes) if raw_classes is not None else ckpt["model_state_dict"]["head.bias"].shape[0]

# Convert classes to Python list
if raw_classes is None:
    classes = [f"class_{{i}}" for i in range(n_old)]
elif isinstance(raw_classes, np.ndarray):
    classes = raw_classes.tolist()
else:
    classes = list(raw_classes)
# Recreate and load your original BottleneckMLP
class BottleneckMLP(nn.Module):
    def __init__(self, in_dim, hid_dim, lat_dim, n_cls):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, lat_dim),
            nn.ReLU()
        )
        self.head = nn.Linear(lat_dim, n_cls)

    def forward(self, x):
        z = self.encoder(x)
        logits = self.head(z)
        return z, logits

# Extract dims from checkpoint
in_dim     = ckpt["model_state_dict"]["encoder.0.weight"].shape[1]
hid_dim    = ckpt["model_state_dict"]["encoder.0.weight"].shape[0]
lat_dim    = ckpt["model_state_dict"]["encoder.2.weight"].shape[0]

# Instantiate and load weights
bottleneck_model = BottleneckMLP(in_dim, hid_dim, lat_dim, n_old).to(device)
bottleneck_model.load_state_dict(ckpt["model_state_dict"])
bottleneck_model.eval()


# %% [code] In[25]

# %% [code] In[26]
raw_classes

# %% [code] In[27]
bottleneck_model.eval()
with torch.no_grad():
    # Option A: get both z and logits
    z_null, logits_null = bottleneck_model(null_tensor)

    # Option B: just run the encoder if you don’t care about the head
    # z_null = model.encoder(null_tensor)

# 3) Pull back to numpy and squeeze off the batch‐dim
z_null = z_null.cpu().numpy().squeeze(0)  # shape: (LATENT_DIM,)

print("Null bottleneck code:", z_null)
null_label = "<null_val>[sep]<null_val>"

# %% [code] In[28]
idx = 0

# 2) grab the weight‐row (and optional bias)
with torch.no_grad():
    w_cls = bottleneck_model.head.weight[idx]     # torch.Tensor of shape (latent_dim,)
    b_cls = bottleneck_model.head.bias[idx]       # scalar

# %% [code] In[29]
# All motions in each day - description with LLM?
# Intra-trajectory
# Inter-trajectory - summary of transition pattern
gdf_lines

# %% [code] In[30]
# Involvement in each day, transition from POI to POI, what pattern in each stay

nearest_15x10.head(20)

# %% [code] In[31]
import pandas as pd
import numpy as np
import torch

SEP  = "[sep]"
NULL = "<null_val>"
def norm_token(x):
    # robust: works for NaN/None/numbers/strings
    if pd.isna(x):
        return NULL
    s = str(x).strip()
    return s if s else NULL

# build class_key safely (vectorized)
nearest_15x10 = nearest_15x10.copy()
nearest_15x10["class_key"] = (
    nearest_15x10["top_category"].apply(norm_token) + 
    SEP + 
    nearest_15x10["sub_category"].apply(norm_token)
)

# (optional but recommended) normalize raw_classes the same way to avoid mismatches
def normalize_key(k):
    if isinstance(k, str):
        parts = k.split(SEP)
        t = norm_token(parts[0] if len(parts)>0 else None)
        s = norm_token(parts[1] if len(parts)>1 else None)
        return f"{t}{SEP}{s}"
    return k

raw_classes_norm = [normalize_key(k) for k in raw_classes]

# lookup index for each key
cls_to_idx = {k:i for i,k in enumerate(raw_classes_norm)}
nearest_15x10["class_idx"] = nearest_15x10["class_key"].map(cls_to_idx).astype("Int64")

# attach 64-D class vectors from model.head.weight
W = bottleneck_model.head.weight.detach().cpu().numpy()                  # (C,64)
W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)    # normalize

nearest_15x10["class_vec_64"] = nearest_15x10["class_idx"].apply(
    lambda idx: None if pd.isna(idx) else W[int(idx)].astype(np.float32)
)

# %% [code] In[32]
nearest_15x10

