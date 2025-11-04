# ─── Standard & Third-Party Libraries ──────────────────────────────────────────
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
from new_pipeline.aggregation import mean_vectors_by_group
from new_pipeline.embedding import load_finetuned_model
from new_pipeline.fill import compute_placeholder_latent, fill_missing_vectors
import pyarrow.csv as pv
from pyarrow.csv import ReadOptions, ParseOptions
# ─── Configuration ───────────────────────────────────────────────────────────

CONFIG = {
    # File Paths
    "HEX_FILE_PATH": "Hex_tesse_raw.parquet",
    "POI_FILE_PATH": "Hull_FL_poi_vec_subset.csv",
    "CHECKPOINT_PATH": "bottleneck_mlp_newdata.pth",
    "OUTPUT_PATH": "POI_encoded_embeddings.parquet",
    "LLM_CHECKPOINT_PATH": None,
    "BASE_MODEL": "google/gemma-3-1b-it",
    
    # Coordinate Reference Systems
    "CRS_GEOGRAPHIC": "4326",
    "CRS_PROJECTED": "5070",  # Using an equal-area projection for the US
    
    # Model Hyperparameters
    "LATENT_DIM": 64,
    "HIDDEN_DIM": 256,
    
    # Training Hyperparameters
    "BATCH_SIZE": 128,
    "LEARNING_RATE": 1e-4,
    "EPOCHS": 100,
    
    # System Configuration
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "NUM_WORKERS": 4,
}

# ─── PyTorch Model Definition ────────────────────────────────────────────────

class BottleneckMLP(nn.Module):
    """A Bottleneck Multi-Layer Perceptron for dimensionality reduction and classification."""
    def __init__(self, in_dim, hid_dim, lat_dim, n_cls):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(hid_dim, lat_dim),
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.head = nn.Linear(lat_dim, n_cls)

    def forward(self, x):
        z = self.encoder(x)
        logits = self.head(z)
        return z, logits

# ─── Data Loading and Preprocessing Functions ────────────────────────────────

def load_hexagon_data(file_path, crs):
    """Loads and preprocesses the hexagon GeoDataFrame."""
    logging.info(f"Loading hexagon data from {file_path}...")
    hex_gdf = pd.read_parquet(file_path)
    hex_gdf = hex_gdf.reset_index(drop=True)
    hex_gdf["hex_id"] = hex_gdf.index.astype(str)
    hex_gdf["geometry"] = gpd.GeoSeries.from_wkb(hex_gdf["geometry"])
    hex_gdf = gpd.GeoDataFrame(hex_gdf, geometry="geometry", crs=crs)
    logging.info(f"Hexagon data loaded with {len(hex_gdf)} hexagons.")
    return hex_gdf

def load_poi_data(file_path, crs):
    """Loads and preprocesses the POI GeoDataFrame."""
    logging.info(f"Loading POI data from {file_path}...")
    table = pv.read_csv(
        file_path,
        read_options=ReadOptions(block_size=1 << 20),
        parse_options=ParseOptions(delimiter=",", quote_char='"', newlines_in_values=True)
    )
    df = table.to_pandas()
    df["geometry"] = df["geometry"].apply(wkt.loads)
    poi_gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=crs)
    logging.info(f"POI data loaded with {len(poi_gdf)} points.")
    return poi_gdf

def parse_vector_column(series: pd.Series) -> np.ndarray:
    """Parses a string representation of vectors into a stacked NumPy array."""
    logging.info("Parsing string vectors into NumPy array...")
    def parse_vec(s: str) -> np.ndarray:
        if isinstance(s, (list, np.ndarray)):
            return np.array(s, dtype=np.float32)
        return np.fromstring(s.strip("[]"), sep=" ", dtype=np.float32)
    
    vecs = np.stack(series.map(parse_vec).values)
    return vecs

# ─── Model Training and Inference Functions ──────────────────────────────────

def train_or_load_model(config, loader, n_classes, class_labels):
    """Instantiates the model and optimizer, then loads from checkpoint or trains."""
    logging.info("Initializing model, optimizer, and criterion...")
    model = BottleneckMLP(
        in_dim=loader.dataset.tensors[0].shape[1],
        hid_dim=config["HIDDEN_DIM"],
        lat_dim=config["LATENT_DIM"],
        n_cls=n_classes
    ).to(config["DEVICE"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.get("EPOCHS", 100),  # defaults to 100 if absent
        eta_min=0.0
    )
    criterion = nn.CrossEntropyLoss()

    if os.path.exists(config["CHECKPOINT_PATH"]):
        logging.info(f"Loading pretrained model from {config['CHECKPOINT_PATH']}")
        ckpt = torch.load(config["CHECKPOINT_PATH"], map_location=config["DEVICE"],weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    else:
        logging.info("No checkpoint found—starting training from scratch.")
        for epoch in range(1, config["EPOCHS"] + 1):
            model.train()
            loop = tqdm(loader, desc=f"Epoch {epoch}/{config['EPOCHS']}", unit="batch")
            total_loss = 0.0
            for xb, yb in loop:
                xb, yb = xb.to(config["DEVICE"]), yb.to(config["DEVICE"])
                optimizer.zero_grad()
                _, logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)
                loop.set_postfix(loss=loss.item())
            avg_loss = total_loss / len(loader.dataset)
            scheduler.step()
            print(f"→ Epoch {epoch:2d}: avg loss = {avg_loss:.4f}")
        
        logging.info(f"Training complete—saving checkpoint to {config['CHECKPOINT_PATH']}")
        torch.save({
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "classes":              class_labels
        }, "bottleneck_mlp_newdata.pth")
    
    return model

def encode_features(model, loader, device):
    """Runs inference to generate latent embeddings for the input data."""
    logging.info("Encoding features to generate latent vectors (Z)...")
    model.eval()
    all_z = []
    with torch.no_grad():
        for xb, _ in tqdm(loader, desc="Encoding"):
            xb = xb.to(device)
            z = model.encoder(xb)
            all_z.append(z.cpu().numpy())
    
    return np.vstack(all_z)

# ─── Geospatial Processing Function ──────────────────────────────────────────

def assign_pois_to_hexagons(poi_gdf, hex_gdf):
    """
    Assign POIs to hexagons and compute average embeddings per cell.

    Returns a GeoDataFrame with one row per hexagon containing the averaged
    embedding vector and POI counts.
    """
    logging.info("Reprojecting GeoDataFrames to equal-area CRS for accurate nearest-neighbor search...")
    poi_proj = poi_gdf.to_crs(epsg=CONFIG["CRS_PROJECTED"])
    hex_proj = hex_gdf.to_crs(epsg=CONFIG["CRS_PROJECTED"])

    logging.info("Assigning POIs to nearest hexagon...")
    joined_gdf = gpd.sjoin_nearest(
        poi_proj[["label_pair", "z", "geometry"]],
        hex_proj[["hex_id", "geometry"]],
        how="left",
    )

    matched = joined_gdf["hex_id"].notna().sum()
    logging.info("Join completed. Matched points: %s/%s", matched, len(poi_gdf))

    aggregated = mean_vectors_by_group(joined_gdf, "hex_id", "z")
    aggregated = aggregated.rename(columns={"z": "embedding"})
    aggregated["embedding"] = aggregated["embedding"].apply(lambda arr: arr.tolist())

    hex_with_embeddings = hex_proj.merge(aggregated, on="hex_id", how="left")
    hex_with_embeddings = hex_with_embeddings.to_crs(epsg=CONFIG["CRS_GEOGRAPHIC"])
    return hex_with_embeddings

# ─── Main Execution ──────────────────────────────────────────────────────────

def main():
    """Main function to orchestrate the entire workflow."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info(f"Using device: {CONFIG['DEVICE']}")
    
    # 1. Load Data
    hex_gdf = load_hexagon_data(CONFIG["HEX_FILE_PATH"], CONFIG["CRS_GEOGRAPHIC"])
    poi_gdf = load_poi_data(CONFIG["POI_FILE_PATH"], CONFIG["CRS_GEOGRAPHIC"])
    print(poi_gdf['concatenated'])
    sep = "[sep]"
    poi_gdf['label_pair'] = (
        poi_gdf['concatenated']
            .str
            .split(sep, regex=False)   # ← literal split
            .str[:2]                   # take the first two pieces
            .str.join(sep)             # re-join them with “[sep]”
    )
    print(poi_gdf['label_pair'])
    # 2. Prepare Data for PyTorch
    X_vecs = parse_vector_column(poi_gdf["concatenated_vec"])
    X_t = torch.from_numpy(X_vecs)
    
    # Create labels for the bottleneck training task
    #poi_gdf['label_pair'] = poi_gdf['concatenated'].str.split('[sep]').str[:2].str.join('[sep]')
    le = LabelEncoder().fit(poi_gdf['label_pair'])
    y_t = torch.from_numpy(le.transform(poi_gdf["label_pair"].values)).long()
    
    pytorch_dataset = TensorDataset(X_t, y_t)
    data_loader = DataLoader(
        pytorch_dataset, 
        batch_size=CONFIG["BATCH_SIZE"], 
        shuffle=False, # Shuffle should be True for training, False for deterministic encoding
        num_workers=CONFIG["NUM_WORKERS"], 
        pin_memory=True
    )
    
    # 3. Train or Load Model
    model = train_or_load_model(CONFIG, data_loader, len(le.classes_), le.classes_)
    
    # 4. Encode Features to get latent vectors
    # Use a new loader with shuffle=False for predictable output order
    encoding_loader = DataLoader(pytorch_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, num_workers=CONFIG["NUM_WORKERS"])
    Z_vectors = encode_features(model, encoding_loader, CONFIG["DEVICE"])
    poi_gdf["z"] = list(Z_vectors)
    
    # 5. Spatially Join POIs (with new embeddings) to Hexagons
    final_joined_gdf = assign_pois_to_hexagons(poi_gdf, hex_gdf)

    # 5b. Fill missing embeddings with placeholder latent
    fallback_latent = np.zeros(CONFIG["LATENT_DIM"], dtype=np.float32)
    if CONFIG.get("LLM_CHECKPOINT_PATH"):
        try:
            tokenizer, llm_model, llm_device = load_finetuned_model(
                CONFIG["LLM_CHECKPOINT_PATH"],
                CONFIG["BASE_MODEL"],
                device=CONFIG["DEVICE"],
            )
            fallback_latent = compute_placeholder_latent(
                tokenizer,
                llm_model,
                llm_device,
                model,
            )
        except Exception as exc:
            logging.warning("Failed to compute placeholder latent via LLM: %s", exc)

    if "embedding" in final_joined_gdf.columns:
        final_joined_gdf["embedding"] = fill_missing_vectors(
            final_joined_gdf["embedding"], fallback_latent
        )

    # 6. Save Final Results
    logging.info(f"Saving final joined data to {CONFIG['OUTPUT_PATH']}...")
    final_joined_gdf.to_parquet(CONFIG["OUTPUT_PATH"])
    logging.info("Script finished successfully.")
    print(final_joined_gdf.head)

if __name__ == "__main__":
    main()
