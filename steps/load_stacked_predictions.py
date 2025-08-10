"""
Step to load stacked DeepSurv predictions.
"""

import os
import json
import h5py
import numpy as np
from typing import Dict, Tuple, Annotated
from zenml import step


@step
def load_stacked_predictions() -> Tuple[
    Annotated[Dict[str, np.ndarray], "deepsurv_temporal_predictions"],
    Annotated[Dict[str, np.ndarray], "deepsurv_spatial_predictions"]
]:
    """
    Load stacked DeepSurv predictions from H5 files.
    
    Returns:
        Tuple of dictionaries containing temporal and spatial predictions.
        Each dictionary has group IDs as keys and predictions as values.
        Predictions have shape (2, 5, n_samples).
    """
    stacked_dir = "results/final_deploy/stacked_predictions"
    summary_path = os.path.join(stacked_dir, "stacking_summary.json")
    
    # Load summary
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    temporal_predictions = {}
    spatial_predictions = {}
    
    print("Loading stacked DeepSurv predictions...")
    
    # Load temporal predictions
    for group_id, info in summary["temporal"].items():
        with h5py.File(info["path"], 'r') as f:
            predictions = f['predictions'][:]
            temporal_predictions[group_id] = predictions
        print(f"  Loaded {group_id} temporal: {predictions.shape}")
    
    # Load spatial predictions
    for group_id, info in summary["spatial"].items():
        with h5py.File(info["path"], 'r') as f:
            predictions = f['predictions'][:]
            spatial_predictions[group_id] = predictions
        print(f"  Loaded {group_id} spatial: {predictions.shape}")
    
    print(f"\nLoaded {len(temporal_predictions)} DeepSurv groups")
    
    return temporal_predictions, spatial_predictions