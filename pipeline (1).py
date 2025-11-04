#!/usr/bin/env python3
import logging
import math
import os
import ast
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, MultiPoint, Polygon
from shapely.ops import unary_union

import libpysal
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE          # ← Added this import

import pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ─── Helpers ─────────────────────────────────────────────────────────────
def read_rg(path, columns=None):
    pf = pq.ParquetFile(path)
    parts = [
        pf.read_row_group(i, columns=columns).to_pandas()
        for i in range(pf.num_row_groups)
    ]
    return pd.concat(parts, ignore_index=True)


def parse_coord_string(s):
    """Safely parse '(lat, lon)' into tuple of floats or (nan, nan)."""
    if not isinstance(s, str):
        if isinstance(s, (tuple, list)) and len(s) == 2:
            try:
                return float(s[0]), float(s[1])
            except:
                return (np.nan, np.nan)
        return (np.nan, np.nan)
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (tuple, list)) and len(parsed) == 2:
            return float(parsed[0]), float(parsed[1])
    except Exception:
        pass
    return (np.nan, np.nan)

def embed_texts(texts, tokenizer, model, device):
    """Batch-tokenize & get final non-pad token embeddings."""
    clean = ["" if t is None else str(t).replace("[sep]", tokenizer.sep_token)
             for t in texts]
    enc = tokenizer(
        clean,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
        return_token_type_ids=False,
    ).to(device)

    with torch.no_grad():
        out = model.base_model(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            output_hidden_states=True
        )
    last_hid = out.hidden_states[-1]  # (B, L, H)
    seq_lens = enc.attention_mask.sum(dim=1) - 1
    embs = last_hid[torch.arange(len(clean)), seq_lens, :]
    return embs

# ─── Main Logic ─────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 1) Load & parse your hurricane data → df_long
    df = pd.read_csv("hurricane_matrix.csv").rename(columns={"Unnamed: 0": "traj_id"})
    df = df.set_index("traj_id").stack().rename("coords").reset_index().rename(columns={"level_1":"pt_idx"})
    df["coords"] = df["coords"].apply(parse_coord_string)
    df[["latitude","longitude"]] = pd.DataFrame(df["coords"].tolist(), index=df.index)
    df["geometry"] = [Point(xy) for xy in zip(df.longitude, df.latitude)]
    point_gdf = gpd.GeoDataFrame(df.drop(columns="coords"), geometry="geometry", crs="EPSG:4326")
    logging.info(f"Built point_gdf with {len(point_gdf)} points")

    # 2) Build concave hull
    coords = [(x,y) for x,y in zip(df.longitude, df.latitude)]
    multip = MultiPoint(coords)
    hull = gpd.GeoSeries([multip]).concave_hull(ratio=0.05, allow_holes=False).iloc[0]
    logging.info("Computed concave hull.")

    # 3) Load hex grid and POI, filter by hull
    hex_gdf = pd.read_parquet("Hex_tesse_raw.parquet")
    hex_gdf["geometry"] = gpd.GeoSeries.from_wkb(hex_gdf["geometry"])
    hex_gdf = gpd.GeoDataFrame(hex_gdf, geometry="geometry", crs="EPSG:4326")
    hex_gdf["hex_id"] = hex_gdf.index.astype(str)

    poi_df = pd.read_csv("Hex_bound_POI.csv")
    poi_gdf = gpd.GeoDataFrame(
        poi_df,
        geometry=gpd.points_from_xy(poi_df.LONGITUDE, poi_df.LATITUDE),
        crs="EPSG:4326"
    )
    poi_gdf["concatenated"] = (
        poi_gdf["concatenated"]
            .str.split(r"\[sep\]").str[:2].str.join("[sep]")
    )
    poi_gdf = poi_gdf[poi_gdf.within(hull)].reset_index(drop=True)
    logging.info(f"Filtered to {len(poi_gdf)} POIs within hull.")

    # 4) Load your fine-tuned LLM + tokenizer
    CHECKPOINT = "/storage1/fs1/nlin/Active/sizhe/FO_DATA/checkpoint-dir/checkpoint-551"
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    base = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it", attn_implementation="eager").to(device)
    model = PeftModel.from_pretrained(base, CHECKPOINT).to(device).eval()
    logging.info("Model & tokenizer loaded.")

    # 5) Embed each unique label
    unique = poi_gdf["concatenated"].unique().tolist()
    rows = []
    for txt in tqdm(unique, desc="Embedding labels"):
        emb = embed_texts([txt], tokenizer, model, device)[0].cpu().numpy()
        rows.append({"label": txt, "embedding": emb})
    df_labels = pd.DataFrame(rows)
    logging.info("Label embeddings complete.")



    # ── 1) HYPERPARAMETERS ─────────────────────────────────────────────────────────
    LATENT_DIM   = 64     # target lower dimension
    HIDDEN_DIM   = 256     # hidden size of MLP
    BATCH_SIZE   = 128
    LR           = 1e-4
    EPOCHS       = 50
    DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_cols = [c for c in pq.ParquetFile("POI_vec_proj_matrix.parquet").schema.names if c != "geometry"]
    vec_proj = read_rg("POI_vec_proj_matrix.parquet", columns=all_cols)
    vec_proj["category"] = vec_proj["concatenated"].str.split(r"\[sep\]").str[0]
    vec_proj["subcategory"] = vec_proj["concatenated"].str.split(r"\[sep\]").str[1]
    # 1.1 make a new string label
    vec_proj['label_pair'] = (
        vec_proj['category']
        + '[sep]'
        + vec_proj['subcategory']
    )
    vec_proj["vec_arr"] = vec_proj["concatenated_vec"].apply(lambda s: np.fromstring(s.strip("[]"), sep=" "))
    logging.info("Loaded POI projection matrix.")
    
    # ── 2) PREPARE DATA ────────────────────────────────────────────────────────────
    # Extract X matrix and integer labels y

    # PyTorch Dataset
    class EmbedDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.from_numpy(X).float()
            self.y = torch.from_numpy(y).long()
        def __len__(self):
            return len(self.y)
        def __getitem__(self, i):
            return self.X[i], self.y[i]

    ds    = EmbedDataset(X, y)
    loader= DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)

    # ── 3) MODEL DEFINITION ────────────────────────────────────────────────────────
    class BottleneckMLP(nn.Module):
        def __init__(self, in_dim, hid_dim, lat_dim, n_cls):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(in_dim, hid_dim),
                nn.LeakyReLU(0.01,True),
                nn.Linear(hid_dim, lat_dim),
                nn.LeakyReLU(0.01,True)
            )
            self.head = nn.Linear(lat_dim, n_cls)
        def forward(self, x):
            z      = self.encoder(x)
            logits = self.head(z)
            return z, logits

    model = BottleneckMLP(
        in_dim=X.shape[1],
        hid_dim=HIDDEN_DIM,
        lat_dim=LATENT_DIM,
        n_cls=n_classes
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # ── 4) TRAIN ─────────────────────────────────────────────────────────────────
    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            _, logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        print(f"Epoch {epoch:2d}  loss = {total_loss/len(ds):.4f}")

    # ── 5) EXTRACT LATENT CODES & APPEND TO df ────────────────────────────────────
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X).float().to(DEVICE)
        Z_tensor, _ = model(X_tensor)               # (N, LATENT_DIM)
        Z = Z_tensor.cpu().numpy()

    # Build a small DataFrame of the new features
    #cols_z   = [f"z_{i}" for i in range(LATENT_DIM)]
    #df_z      = pd.DataFrame(Z, columns=cols_z, index=df_labels.index)
    print("Training completed!")
    # Concatenate back onto your original df
    #df_supervised = pd.concat([vec_proj, df_z], axis=1)
    checkpoint = {
    'epoch': epoch,                   # last epoch you trained
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()}

    # 2) Write it out
    torch.save(checkpoint, "bottleneck_mlp_checkpoint.pth")
    print("Saving completed!")
    # 7) Read in POI vector projections & parse into arrays

    # 8) Batch-encode your POIs into z_poi
    batch_size = 1000
    Z_chunks = []
    with torch.no_grad():
        for start in tqdm(range(0, len(vec_proj), batch_size), desc="Embedding POIs"):
            batch = torch.from_numpy(
                np.stack(vec_proj["vec_arr"].iloc[start:start+batch_size].values)
            ).float().to(device)
            z_b = model.base_model.encoder(batch) if hasattr(model, "base_model") else model.encoder(batch)
            Z_chunks.append(z_b.cpu().numpy())
    Z = np.vstack(Z_chunks)
    vec_proj["z_poi"] = list(Z)
    vec_proj.to_parquet("POI_vec_proj_matrix.parquet")


    # 9) t-SNE on top 10 categories & plot
    #vec_proj["category"] = vec_proj["concatenated"].str.split(r"\[sep\]").str[0]
    top10 = vec_proj["category"].value_counts().nlargest(10).index
    sub = vec_proj[vec_proj["category"].isin(top10)].copy()
    per_cat = 200  # e.g. 200 × 10 = 2 000 total
    sub_sampled = (
        sub
        .groupby("category", group_keys=False)
        .apply(lambda df: df.sample(n=min(len(df), per_cat),
                                    random_state=42))
        .reset_index(drop=True)
    )
    Z_sub = np.stack(sub_sampled["z_poi"].values)

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="random",
        random_state=42
    )
    Z2 = tsne.fit_transform(Z_sub)

    # and then use `sub_sampled` (instead of `sub`) for plotting
    sub = sub_sampled

    sub["cat_code"] = sub["category"].astype("category").cat.codes
    cats = sub["category"].astype("category").cat.categories

    plt.figure(figsize=(10, 8))
    sc = plt.scatter(Z2[:, 0], Z2[:, 1], c=sub["cat_code"], alpha=0.7)
    plt.title("t-SNE of Supervised Latents (Top 10 Categories)")
    plt.xlabel("TSNE-1"); plt.ylabel("TSNE-2")

    handles = [
        plt.Line2D([], [], marker="o", linestyle="", 
                   color=sc.cmap(sc.norm(i)), label=cat)
        for i, cat in enumerate(cats)
    ]
    plt.legend(handles=handles, title="Category",
               bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    plt.tight_layout()

    # ← saves to disk
    plt.savefig("Fullset_Cleaned_tsne.png", dpi=300, bbox_inches="tight")
    logging.info("Saved t-SNE plot as Cleaned_tsne.png")

    # ← and displays inline (e.g. in Jupyter)
    plt.show()

if __name__ == "__main__":
    main()
