"""
Step to load DeepHit predictions.
"""

import os
import json
import h5py
import numpy as np
from typing import Dict, Tuple, Annotated
from zenml import step


@step
def load_deephit_predictions() -> Tuple[
    Annotated[Dict[str, np.ndarray], "deephit_temporal_predictions"],
    Annotated[Dict[str, np.ndarray], "deephit_spatial_predictions"]
]:
    """
    Load DeepHit predictions from H5 files.
    
    Returns:
        Tuple of dictionaries containing temporal and spatial predictions.
        Each dictionary has model names as keys and predictions as values.
        Predictions have shape (2, 5, n_samples).
    """
    predictions_dir = "results/final_deploy/individual_predictions"
    summary_path = os.path.join(predictions_dir, "prediction_generation_summary_20250808_113457.json")
    
    # Load summary
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    temporal_predictions = {}
    spatial_predictions = {}
    
    print("\nLoading DeepHit predictions...")
    
    # Filter DeepHit models (models 25-36)
    deephit_models = [m for m in summary["results"] if m["model_type"] == "deephit"]
    
    for model_info in deephit_models:
        model_name = model_info["model_name"]
        
        # Load temporal predictions
        with h5py.File(model_info["temporal_predictions_path"], 'r') as f:
            predictions = f['predictions'][:]
            temporal_predictions[model_name] = predictions
        print(f"  Loaded {model_name} temporal: {predictions.shape}")
        
        # Load spatial predictions
        with h5py.File(model_info["spatial_predictions_path"], 'r') as f:
            predictions = f['predictions'][:]
            spatial_predictions[model_name] = predictions
        print(f"  Loaded {model_name} spatial: {predictions.shape}")
    
    print(f"\nLoaded {len(temporal_predictions)} DeepHit models")
    
    return temporal_predictions, spatial_predictions