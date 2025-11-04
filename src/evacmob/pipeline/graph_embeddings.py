"""Graph embedding utilities (Node2Vec) for spatial surfaces."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
from gensim.models import Word2Vec
from node2vec import Node2Vec

LOGGER = logging.getLogger(__name__)


@dataclass
class Node2VecConfig:
    """Configuration for building Node2Vec embeddings."""

    dimensions: int = 64
    walk_length: int = 128
    num_walks: int = 16
    window: int = 16
    min_count: int = 3
    workers: int = 4
    p: float = 1.0
    q: float = 1.0
    epochs: int = 1
    seed: int = 42
    temp_folder: Path | None = None


def build_graph_from_geodataframe(
    gdf: gpd.GeoDataFrame,
    id_column: str,
    weight_column: str | None = None,
) -> nx.Graph:
    """
    Construct a graph by considering polygon contiguity.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input polygons (e.g., hexagons or block groups). Must be in projected CRS.
    id_column : str
        Unique identifier column for nodes.
    weight_column : str, optional
        If provided, assign edge weights based on this column (e.g., shared boundary length).
    """
    if gdf.crs is None:
        raise ValueError("GeoDataFrame must have a CRS before building a graph.")

    gdf = gdf[[id_column, "geometry"]].copy()
    gdf["geometry"] = gdf.geometry.buffer(0)  # clean topology
    neighbors = gpd.sjoin(gdf, gdf, how="left", predicate="touches")

    G = nx.Graph()
    for node in gdf[id_column]:
        G.add_node(node)

    for _, row in neighbors.iterrows():
        src = row[id_column + "_left"]
        dst = row[id_column + "_right"]
        if src == dst:
            continue
        weight = None
        if weight_column:
            weight = row.get(weight_column)
        if G.has_edge(src, dst):
            continue
        G.add_edge(src, dst, weight=weight)

    LOGGER.info("Constructed graph with %s nodes and %s edges", G.number_of_nodes(), G.number_of_edges())
    return G


def run_node2vec(
    graph: nx.Graph,
    config: Node2VecConfig,
) -> Tuple[Word2Vec, Dict[str, np.ndarray]]:
    """
    Fit a Node2Vec model on the given graph and return embeddings.
    """
    node2vec = Node2Vec(
        graph,
        dimensions=config.dimensions,
        walk_length=config.walk_length,
        num_walks=config.num_walks,
        workers=config.workers,
        p=config.p,
        q=config.q,
        seed=config.seed,
        temp_folder=str(config.temp_folder) if config.temp_folder else None,
    )
    model = node2vec.fit(
        vector_size=config.dimensions,
        window=config.window,
        min_count=config.min_count,
        workers=config.workers,
        epochs=config.epochs,
        seed=config.seed,
    )
    embeddings = {node: model.wv[node] for node in graph.nodes()}
    return model, embeddings


def save_embeddings(
    model: Word2Vec,
    embeddings: Dict[str, np.ndarray],
    text_path: Path,
    model_path: Path,
) -> None:
    """Persist embeddings to disk for later reuse."""
    model.wv.save_word2vec_format(str(text_path))
    model.save(str(model_path))
    LOGGER.info("Saved embeddings to %s and model to %s", text_path, model_path)


def embeddings_to_dataframe(embeddings: Dict[str, np.ndarray]) -> gpd.GeoDataFrame:
    """Convert embeddings dict to a tabular representation."""
    return gpd.GeoDataFrame(embeddings)
