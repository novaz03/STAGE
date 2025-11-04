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


mean_vecs = (
    joined
    .groupby('hex_id')['concatenated_vec']
    .agg(lambda arrs: np.mean(np.stack(arrs.values), axis=0))
)
# mean_vecs is a Series: index=hex_id, value=np.ndarray(1152,)

# 2a) merge into hex_gdf
hex_gdf = hex_gdf.merge(
    mean_vecs.rename('vec_mean'),
    left_on='hex_id',
    right_index=True,
    how='left'
)