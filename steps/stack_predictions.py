"""
Stack Predictions Step for Ensemble Deployment

This module contains the ZenML step for stacking predictions from multiple models,
particularly grouping DeepSurv predictions by event type.
"""

import os
import h5py
import numpy as np
from zenml.steps import step
from typing import List, Dict, Any, Tuple
import pandas as pd


@step(enable_cache=True)
def stack_predictions(
    deployment_details: List[Dict[str, Any]],
    temporal_predictions_paths: List[str],
    spatial_predictions_paths: List[str]
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Stack predictions from multiple models according to the ensemble strategy.
    
    For DeepSurv models:
    - Group by (Algorithm, Structure, Balancing Method, Optimization Target)
    - Stack Event 1 and Event 2 predictions together
    
    For DeepHit models:
    - Already in correct format (2 events x 5 time points x n_samples)
    
    Args:
        deployment_details: List of deployment details for each model
        temporal_predictions_paths: List of paths to temporal prediction files
        spatial_predictions_paths: List of paths to spatial prediction files
        
    Returns:
        Tuple of (temporal_stacked_predictions, spatial_stacked_predictions)
        Each is a dictionary with model group identifiers as keys
    """
    print("\n=== Stacking Predictions ===")
    
    # Separate DeepSurv and DeepHit models
    deepsurv_models = []
    deephit_models = []
    
    for i, details in enumerate(deployment_details):
        model_config = details.get('model_config', {})
        if model_config.get('algorithm', '').lower() == 'deepsurv':
            deepsurv_models.append({
                'index': i,
                'config': model_config,
                'details': details,
                'temporal_path': temporal_predictions_paths[i],
                'spatial_path': spatial_predictions_paths[i]
            })
        else:  # DeepHit
            deephit_models.append({
                'index': i,
                'config': model_config,
                'details': details,
                'temporal_path': temporal_predictions_paths[i],
                'spatial_path': spatial_predictions_paths[i]
            })
    
    print(f"Found {len(deepsurv_models)} DeepSurv models and {len(deephit_models)} DeepHit models")
    
    # Group DeepSurv models
    deepsurv_groups = {}
    for model in deepsurv_models:
        config = model['config']
        # Create group key based on characteristics
        group_key = (
            config.get('algorithm'),
            config.get('structure'),
            config.get('balancing_method'),
            config.get('optimization_target')
        )
        
        if group_key not in deepsurv_groups:
            deepsurv_groups[group_key] = []
        deepsurv_groups[group_key].append(model)
    
    print(f"\nDeepSurv models grouped into {len(deepsurv_groups)} groups:")
    for group_key, models in deepsurv_groups.items():
        print(f"  {group_key}: {len(models)} models")
    
    # Stack predictions
    temporal_stacked = {}
    spatial_stacked = {}
    
    # Process DeepSurv groups
    for group_idx, (group_key, models) in enumerate(deepsurv_groups.items()):
        group_id = f"deepsurv_group_{group_idx}"
        print(f"\nProcessing {group_id}: {group_key}")
        
        # Find Event 1 and Event 2 models
        event1_model = None
        event2_model = None
        
        for model in models:
            endpoint = model['config'].get('prediction_endpoint')
            if endpoint == 'Event 1':
                event1_model = model
            elif endpoint == 'Event 2':
                event2_model = model
        
        if event1_model and event2_model:
            # Load and stack predictions
            temporal_stacked[group_id] = stack_deepsurv_predictions(
                event1_model['temporal_path'],
                event2_model['temporal_path']
            )
            spatial_stacked[group_id] = stack_deepsurv_predictions(
                event1_model['spatial_path'],
                event2_model['spatial_path']
            )
            print(f"  Stacked Event 1 and Event 2 predictions")
        else:
            print(f"  Warning: Missing Event 1 or Event 2 model in group")
    
    # Process DeepHit models (already in correct format)
    for model_idx, model in enumerate(deephit_models):
        model_id = f"deephit_{model_idx}"
        config = model['config']
        print(f"\nProcessing {model_id}: {config.get('structure')} - {config.get('balancing_method')}")
        
        # Load predictions directly
        temporal_stacked[model_id] = load_predictions_from_h5(model['temporal_path'])
        spatial_stacked[model_id] = load_predictions_from_h5(model['spatial_path'])
        print(f"  Loaded DeepHit predictions")
    
    print(f"\n=== Stacking Complete ===")
    print(f"Total stacked models: {len(temporal_stacked)}")
    
    return temporal_stacked, spatial_stacked


def stack_deepsurv_predictions(event1_path: str, event2_path: str) -> np.ndarray:
    """
    Stack DeepSurv predictions for Event 1 and Event 2.
    
    Args:
        event1_path: Path to Event 1 predictions
        event2_path: Path to Event 2 predictions
        
    Returns:
        Stacked predictions with shape (2, 5, n_samples)
    """
    # Load predictions
    event1_pred = load_predictions_from_h5(event1_path)
    event2_pred = load_predictions_from_h5(event2_path)
    
    # Convert DeepSurv survival predictions to CIF if needed
    if event1_pred.ndim == 2:  # (time_points, n_samples)
        # Extract 5 time points: [365, 730, 1095, 1460, 1825]
        time_indices = [364, 729, 1094, 1459, 1824]  # 0-indexed
        
        # Ensure we don't exceed array bounds
        max_idx = event1_pred.shape[0] - 1
        time_indices = [min(idx, max_idx) for idx in time_indices]
        
        event1_cif = convert_survival_to_cif(event1_pred[time_indices, :])
        event2_cif = convert_survival_to_cif(event2_pred[time_indices, :])
        
        # Stack to create (2, 5, n_samples)
        stacked = np.stack([event1_cif, event2_cif], axis=0)
    else:
        # Already in correct format
        stacked = np.stack([event1_pred, event2_pred], axis=0)
    
    return stacked


def convert_survival_to_cif(survival_probs: np.ndarray) -> np.ndarray:
    """
    Convert survival probabilities to cumulative incidence function (CIF).
    
    Args:
        survival_probs: Survival probabilities with shape (n_time_points, n_samples)
        
    Returns:
        CIF with same shape
    """
    # CIF = 1 - Survival probability
    cif = 1.0 - survival_probs
    return cif


def load_predictions_from_h5(file_path: str) -> np.ndarray:
    """
    Load predictions from HDF5 file.
    
    Args:
        file_path: Path to HDF5 file
        
    Returns:
        Predictions array
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Predictions file not found: {file_path}")
    
    with h5py.File(file_path, 'r') as f:
        # Try different possible keys
        if 'predictions' in f:
            predictions = f['predictions'][:]
        elif 'cif' in f:
            predictions = f['cif'][:]
        elif 'survival' in f:
            predictions = f['survival'][:]
        else:
            # Use the first dataset found
            keys = list(f.keys())
            if keys:
                predictions = f[keys[0]][:]
            else:
                raise ValueError(f"No data found in {file_path}")
    
    return predictions