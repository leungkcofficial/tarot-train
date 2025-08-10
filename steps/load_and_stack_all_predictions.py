"""
Step to load all 36 model predictions and stack them according to model grouping.
"""

import numpy as np
import pandas as pd
import json
import glob
import h5py
import os
from typing import Tuple, Annotated, Dict
from zenml import step


@step
def load_and_stack_all_predictions() -> Tuple[
    Annotated[np.ndarray, "temporal_cif_all"],
    Annotated[np.ndarray, "spatial_cif_all"],
    Annotated[Dict, "model_info"]
]:
    """
    Load all 36 model predictions and stack them according to model grouping.
    
    Returns:
        Tuple of:
        - temporal_cif_all: Shape (24, 2, 5, n_samples) - all temporal predictions
        - spatial_cif_all: Shape (24, 2, 5, n_samples) - all spatial predictions
        - model_info: Dictionary with model grouping information
    """
    # Find the latest prediction summary file
    summary_files = glob.glob('results/final_deploy/individual_predictions/prediction_generation_summary_*.json')
    if not summary_files:
        raise FileNotFoundError("No prediction summary files found")
    latest_summary = max(summary_files)  # Get the latest by timestamp in filename
    
    print(f"Loading predictions from: {latest_summary}")
    
    with open(latest_summary, 'r') as f:
        summary = json.load(f)
    
    # Create model name to info mapping
    model_map = {info["model_name"]: info for info in summary["results"]}
    
    # Define the 12 DeepSurv groups based on model_grouping_summary.md
    DEEPSURV_GROUPS = [
        {"name": "Group 1: ANN + None + CI", "event1": "Ensemble_model1_DeepSurv_ANN_Event_1", "event2": "Ensemble_model3_DeepSurv_ANN_Event_2"},
        {"name": "Group 2: ANN + None + LL", "event1": "Ensemble_model2_DeepSurv_ANN_Event_1", "event2": "Ensemble_model4_DeepSurv_ANN_Event_2"},
        {"name": "Group 3: ANN + NearMiss1 + CI", "event1": "Ensemble_model5_DeepSurv_ANN_Event_1", "event2": "Ensemble_model7_DeepSurv_ANN_Event_2"},
        {"name": "Group 4: ANN + NearMiss1 + LL", "event1": "Ensemble_model6_DeepSurv_ANN_Event_1", "event2": "Ensemble_model8_DeepSurv_ANN_Event_2"},
        {"name": "Group 5: ANN + KNN + CI", "event1": "Ensemble_model9_DeepSurv_ANN_Event_1", "event2": "Ensemble_model11_DeepSurv_ANN_Event_2"},
        {"name": "Group 6: ANN + KNN + LL", "event1": "Ensemble_model10_DeepSurv_ANN_Event_1", "event2": "Ensemble_model12_DeepSurv_ANN_Event_2"},
        {"name": "Group 7: LSTM + None + CI", "event1": "Ensemble_model13_DeepSurv_LSTM_Event_1", "event2": "Ensemble_model15_DeepSurv_LSTM_Event_2"},
        {"name": "Group 8: LSTM + None + LL", "event1": "Ensemble_model14_DeepSurv_LSTM_Event_1", "event2": "Ensemble_model16_DeepSurv_LSTM_Event_2"},
        {"name": "Group 9: LSTM + NearMiss3 + CI", "event1": "Ensemble_model17_DeepSurv_LSTM_Event_1", "event2": "Ensemble_model19_DeepSurv_LSTM_Event_2"},
        {"name": "Group 10: LSTM + NearMiss3 + LL", "event1": "Ensemble_model18_DeepSurv_LSTM_Event_1", "event2": "Ensemble_model20_DeepSurv_LSTM_Event_2"},
        {"name": "Group 11: LSTM + KNN + CI", "event1": "Ensemble_model21_DeepSurv_LSTM_Event_1", "event2": "Ensemble_model23_DeepSurv_LSTM_Event_2"},
        {"name": "Group 12: LSTM + KNN + LL", "event1": "Ensemble_model22_DeepSurv_LSTM_Event_1", "event2": "Ensemble_model24_DeepSurv_LSTM_Event_2"}
    ]
    
    # DeepHit models (models 25-36)
    DEEPHIT_MODELS = [f"Ensemble_model{i}_DeepHit_{'ANN' if i <= 30 else 'LSTM'}_Both" for i in range(25, 37)]
    
    # Time points to extract (in days) - for DeepSurv models
    TIME_POINTS = [365, 730, 1095, 1460, 1825]
    
    def load_predictions(h5_path):
        """Load predictions from H5 file."""
        with h5py.File(h5_path, 'r') as f:
            return f['predictions'][:]
    
    def extract_at_time_points(predictions, time_points):
        """Extract DeepSurv predictions at specific time points."""
        indices = [t - 1 for t in time_points]  # 0-based indexing
        return predictions[indices, :]
    
    # Load and stack all predictions
    temporal_cif_matrices = []
    spatial_cif_matrices = []
    model_names = []
    
    # Process DeepSurv groups
    print("\nLoading DeepSurv predictions...")
    for i, group in enumerate(DEEPSURV_GROUPS):
        # Get model info
        event1_info = model_map[group["event1"]]
        event2_info = model_map[group["event2"]]
        
        # Load temporal predictions
        event1_temporal = load_predictions(event1_info["temporal_predictions_path"])
        event2_temporal = load_predictions(event2_info["temporal_predictions_path"])
        
        # Extract at time points
        event1_temporal_extracted = extract_at_time_points(event1_temporal, TIME_POINTS)
        event2_temporal_extracted = extract_at_time_points(event2_temporal, TIME_POINTS)
        
        # Stack to create (2, 5, n_samples)
        temporal_stacked = np.stack([event1_temporal_extracted, event2_temporal_extracted], axis=0)
        temporal_cif_matrices.append(temporal_stacked)
        
        # Load spatial predictions
        event1_spatial = load_predictions(event1_info["spatial_predictions_path"])
        event2_spatial = load_predictions(event2_info["spatial_predictions_path"])
        
        # Extract at time points
        event1_spatial_extracted = extract_at_time_points(event1_spatial, TIME_POINTS)
        event2_spatial_extracted = extract_at_time_points(event2_spatial, TIME_POINTS)
        
        # Stack to create (2, 5, n_samples)
        spatial_stacked = np.stack([event1_spatial_extracted, event2_spatial_extracted], axis=0)
        spatial_cif_matrices.append(spatial_stacked)
        
        model_names.append(group['name'])
        print(f"  Loaded {group['name']}: temporal {temporal_stacked.shape}, spatial {spatial_stacked.shape}")
    
    # Process DeepHit models
    print("\nLoading DeepHit predictions...")
    for model_name in DEEPHIT_MODELS:
        model_info = model_map[model_name]
        
        # Load predictions (already in (2, 5, n_samples) format)
        temporal_pred = load_predictions(model_info["temporal_predictions_path"])
        spatial_pred = load_predictions(model_info["spatial_predictions_path"])
        
        temporal_cif_matrices.append(temporal_pred)
        spatial_cif_matrices.append(spatial_pred)
        
        model_names.append(model_name)
        print(f"  Loaded {model_name}: temporal {temporal_pred.shape}, spatial {spatial_pred.shape}")
    
    # Stack all predictions into (24, 2, 5, n_samples)
    temporal_cif_all = np.stack(temporal_cif_matrices, axis=0)
    spatial_cif_all = np.stack(spatial_cif_matrices, axis=0)
    
    print(f"\nFinal CIF matrices shape:")
    print(f"  Temporal: {temporal_cif_all.shape}")
    print(f"  Spatial: {spatial_cif_all.shape}")
    
    # Save model info
    model_info = {
        'model_names': model_names,
        'deepsurv_groups': DEEPSURV_GROUPS,
        'deephit_models': DEEPHIT_MODELS,
        'time_points': TIME_POINTS,
        'n_models': len(model_names),
        'temporal_shape': list(temporal_cif_all.shape),
        'spatial_shape': list(spatial_cif_all.shape)
    }
    
    # Save model info to JSON
    os.makedirs('results/final_deploy/ensemble_evaluation', exist_ok=True)
    with open('results/final_deploy/ensemble_evaluation/model_stacking_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    return temporal_cif_all, spatial_cif_all, model_info