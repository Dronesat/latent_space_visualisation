import os
import pickle
import numpy as np
from typing import Tuple
from scipy.spatial.distance import cdist
import yaml
import warnings

WORKDIR = os.path.expandvars('${HOME}/myproject/cryodrgn_dataset/CryoDRGN/job002/train_128')
EPOCH = 24
KMEANS = 20

'''Load Results'''  
def load_pickle(pickle_path):
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)

z = load_pickle(f'{WORKDIR}/z.{EPOCH}.pkl')
umap = load_pickle(f'{WORKDIR}/analyze.{EPOCH}/umap.pkl')
kmeans_labels = load_pickle(f'{WORKDIR}/analyze.{EPOCH}/kmeans{KMEANS}/labels.pkl')
kmeans_centers = np.loadtxt(f'{WORKDIR}/analyze.{EPOCH}/kmeans{KMEANS}/centers.txt')

def get_nearest_point( data: np.ndarray, queries: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each point in 'queries', find the closest point in 'data'.
    Returns:
        nearest_points: The closest points in 'data' for each query 
        nearest_indices: The indices in 'data' of the closest points
    """
    # Distance from each queries to each data point
    distances = cdist(queries, data)
    # For each query, find the index of the closest datapoint 
    nearest_indices = np.argmin(distances,axis=1)
    # Select the closest data points using the indices
    nearest_points = data[nearest_indices]
    return nearest_points, nearest_indices

# Get index for on-data cluster center
kmeans_centers, centers_ind = get_nearest_point(z, kmeans_centers)

'''Load Dataset'''
def load_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)
    
def load_config(config_path):
    """
    Load configuration from a file path or return the config object if already loaded.
    Supports .yaml/.yml (recommended) and /.pkl (deprecated)
    """
    if isinstance(config_path, str):
        extension = os.path.splitext(config_path)[-1].lower()
        if extension in {'.yml', '.yaml'}:
            return load_yaml(config_path)
        elif extension == '.pkl':
            warnings.warn(
                "Loading configuration from a .pkl file is deprecated. Please use .yaml instead.",
                DeprecationWarning
            )
            return load_pickle(config_path)
        else:
            raise RuntimeError(f"Unrecognized config extension: {extension}")

config = load_config(f'{WORKDIR}/config.yaml')
print(f'Config: {config}')
