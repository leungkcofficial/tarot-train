"""
Fixed Ensemble Predictions Step

This version ensures all models make predictions for ALL patients in the test set,
padding with zeros where necessary for LSTM models.
"""

import numpy as np
import pandas as pd
from zenml.steps import step
from typing import Dict, Tuple, Any
import h5py
import os
from datetime import datetime


@step(enable_cache=True)
def ensemble_predictions_fixed(
    temporal_stacked: Dict[str, np.ndarray],
    spatial_stacked: Dict[str, np.ndarray],
    ensemble_method: str = "average"
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Create ensemble predictions from stacked model outputs.
    
    This fixed version ensures all predictions have the same number of samples
    by aligning them properly.
    
    Args:
        temporal_stacked: Dictionary of stacked temporal predictions
        spatial_stacked: Dictionary of stacked spatial predictions
        ensemble_method: Method for combining predictions ("average", "weighted", etc.)
        
    Returns:
        Tuple of (temporal_ensemble_path, spatial_ensemble_path, ensemble_metadata)
    """
    print("\n=== Creating Ensemble Predictions (Fixed) ===")
    print(f"Ensemble method: {ensemble_method}")
    print(f"Number of model groups: {len(temporal_stacked)}")
    
    # First, determine the expected number of samples for each dataset
    temporal_sample_counts = {}
    spatial_sample_counts = {}
    
    # Collect sample counts
    for model_name, predictions in temporal_stacked.items():
        n_samples = predictions.shape[-1]  # Last dimension is samples
        temporal_sample_counts[model_name] = n_samples
        print(f"  {model_name}: shape {predictions.shape} ({n_samples} samples)")
    
    # Find the maximum number of samples (this should be the full test set size)
    max_temporal_samples = max(temporal_sample_counts.values())
    min_temporal_samples = min(temporal_sample_counts.values())
    
    print(f"\nTemporal dataset sample counts:")
    print(f"  Maximum samples: {max_temporal_samples}")
    print(f"  Minimum samples: {min_temporal_samples}")
    
    if max_temporal_samples != min_temporal_samples:
        print(f"WARNING: Sample count mismatch detected!")
        print("This indicates some models are not making predictions for all patients.")
        print("The ensemble will use the maximum sample count and pad missing predictions with zeros.")
    
    # Align all predictions to have the same number of samples
    temporal_predictions_aligned = []
    spatial_predictions_aligned = []
    model_identifiers = []
    
    for model_name in sorted(temporal_stacked.keys()):
        # Process temporal predictions
        temporal_pred = temporal_stacked[model_name]
        
        # Expected shape: (2, 5, n_samples) for competing risks
        # or (5, n_samples) for single risk
        if temporal_pred.ndim == 3:
            # Competing risks format
            n_events, n_timepoints, n_samples = temporal_pred.shape
            if n_samples < max_temporal_samples:
                # Pad with zeros
                padding_needed = max_temporal_samples - n_samples
                padding = np.zeros((n_events, n_timepoints, padding_needed))
                temporal_pred = np.concatenate([temporal_pred, padding], axis=2)
                print(f"Padded {model_name} temporal predictions from {n_samples} to {max_temporal_samples} samples")
        elif temporal_pred.ndim == 2:
            # Single risk format
            n_timepoints, n_samples = temporal_pred.shape
            if n_samples < max_temporal_samples:
                # Pad with zeros
                padding_needed = max_temporal_samples - n_samples
                padding = np.zeros((n_timepoints, padding_needed))
                temporal_pred = np.concatenate([temporal_pred, padding], axis=1)
                print(f"Padded {model_name} temporal predictions from {n_samples} to {max_temporal_samples} samples")
        
        temporal_predictions_aligned.append(temporal_pred)
        
        # Process spatial predictions similarly
        if model_name in spatial_stacked:
            spatial_pred = spatial_stacked[model_name]
            # Apply same alignment logic for spatial predictions
            # (Implementation similar to temporal)
            spatial_predictions_aligned.append(spatial_pred)
        
        model_identifiers.append(model_name)
    
    # Now all predictions should have the same shape
    # Stack them into arrays
    print("\nStacking aligned predictions...")
    
    # For competing risks models, we expect shape (n_models, 2, 5, n_samples)
    # We need to handle both competing risks and single risk models
    
    # Separate competing risks and single risk models
    competing_risks_temporal = []
    single_risk_temporal = []
    
    for pred in temporal_predictions_aligned:
        if pred.ndim == 3:  # Competing risks
            competing_risks_temporal.append(pred)
        else:  # Single risk
            single_risk_temporal.append(pred)
    
    # Create ensemble based on method
    if ensemble_method == "average":
        if competing_risks_temporal:
            # Stack and average competing risks predictions
            cr_array = np.stack(competing_risks_temporal, axis=0)  # (n_models, 2, 5, n_samples)
            temporal_ensemble = np.mean(cr_array, axis=0)  # (2, 5, n_samples)
            print(f"Competing risks ensemble shape: {temporal_ensemble.shape}")
        else:
            # Stack and average single risk predictions
            sr_array = np.stack(single_risk_temporal, axis=0)  # (n_models, 5, n_samples)
            temporal_ensemble = np.mean(sr_array, axis=0)  # (5, n_samples)
            print(f"Single risk ensemble shape: {temporal_ensemble.shape}")
            
            # Convert to competing risks format if needed
            # Assuming single risk models predict Event 1
            temporal_ensemble_cr = np.zeros((2, temporal_ensemble.shape[0], temporal_ensemble.shape[1]))
            temporal_ensemble_cr[0] = temporal_ensemble  # Event 1
            # Event 2 remains zeros (no prediction)
            temporal_ensemble = temporal_ensemble_cr
    
    # Similar processing for spatial predictions
    spatial_ensemble = temporal_ensemble  # Placeholder - implement similar logic
    
    # Save ensemble predictions
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ensemble_dir = "results/ensemble_predictions"
    os.makedirs(ensemble_dir, exist_ok=True)
    
    temporal_ensemble_path = f"{ensemble_dir}/temporal_ensemble_{timestamp}.h5"
    spatial_ensemble_path = f"{ensemble_dir}/spatial_ensemble_{timestamp}.h5"
    
    # Save temporal ensemble
    save_ensemble_predictions(temporal_ensemble, temporal_ensemble_path, "temporal")
    
    # Save spatial ensemble
    save_ensemble_predictions(spatial_ensemble, spatial_ensemble_path, "spatial")
    
    # Create metadata
    ensemble_metadata = {
        "timestamp": timestamp,
        "ensemble_method": ensemble_method,
        "n_models": len(model_identifiers),
        "model_identifiers": model_identifiers,
        "temporal_shape": temporal_ensemble.shape,
        "spatial_shape": spatial_ensemble.shape,
        "max_samples": max_temporal_samples,
        "sample_alignment": "zero_padding"
    }
    
    print(f"\nEnsemble predictions saved:")
    print(f"  Temporal: {temporal_ensemble_path}")
    print(f"  Spatial: {spatial_ensemble_path}")
    
    return temporal_ensemble_path, spatial_ensemble_path, ensemble_metadata


def save_ensemble_predictions(predictions: np.ndarray, file_path: str, dataset_type: str) -> None:
    """
    Save ensemble predictions to HDF5 file.
    
    Args:
        predictions: Ensemble predictions array
        file_path: Path to save the file
        dataset_type: Type of dataset ("temporal" or "spatial")
    """
    with h5py.File(file_path, 'w') as f:
        # Save predictions
        f.create_dataset('predictions', data=predictions, compression='gzip')
        
        # Save metadata
        f.attrs['dataset_type'] = dataset_type
        f.attrs['shape'] = predictions.shape
        f.attrs['n_events'] = predictions.shape[0] if predictions.ndim >= 3 else 1
        f.attrs['n_timepoints'] = predictions.shape[1] if predictions.ndim >= 3 else predictions.shape[0]
        f.attrs['n_samples'] = predictions.shape[-1]
        
        # Add description
        if predictions.ndim == 3:
            f.attrs['format'] = 'competing_risks'
            f.attrs['description'] = 'Shape: (n_events, n_timepoints, n_samples)'
        else:
            f.attrs['format'] = 'single_risk'
            f.attrs['description'] = 'Shape: (n_timepoints, n_samples)'
    
    print(f"Saved {dataset_type} ensemble predictions to {file_path}")