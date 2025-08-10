"""
Ensemble Predictions Step for Ensemble Deployment

This module contains the ZenML step for creating ensemble predictions
by combining predictions from multiple models.
"""

import os
import numpy as np
import h5py
from datetime import datetime
from zenml.steps import step
from typing import Dict, Tuple, Any
import pandas as pd


@step(enable_cache=True)
def ensemble_predictions(
    temporal_stacked: Dict[str, np.ndarray],
    spatial_stacked: Dict[str, np.ndarray],
    ensemble_method: str = "average",
    output_dir: str = "results/final_deploy/ensemble_predictions"
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Create ensemble predictions by combining predictions from all models.
    
    Args:
        temporal_stacked: Dictionary of stacked temporal predictions
        spatial_stacked: Dictionary of stacked spatial predictions
        ensemble_method: Method for ensembling ("average", "weighted_average", etc.)
        output_dir: Directory to save ensemble predictions
        
    Returns:
        Tuple of (temporal_ensemble_path, spatial_ensemble_path, ensemble_metadata)
    """
    print("\n=== Creating Ensemble Predictions ===")
    print(f"Ensemble method: {ensemble_method}")
    print(f"Number of model groups: {len(temporal_stacked)}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all predictions into arrays
    temporal_predictions = []
    spatial_predictions = []
    model_identifiers = []
    
    for model_id, predictions in temporal_stacked.items():
        temporal_predictions.append(predictions)
        model_identifiers.append(model_id)
        print(f"  {model_id}: shape {predictions.shape}")
    
    for model_id, predictions in spatial_stacked.items():
        spatial_predictions.append(predictions)
    
    # Convert to numpy arrays
    # All predictions should have shape (2, 5, n_samples) or similar
    temporal_array = np.array(temporal_predictions)  # Shape: (n_models, 2, 5, n_samples)
    spatial_array = np.array(spatial_predictions)
    
    print(f"\nTemporal predictions array shape: {temporal_array.shape}")
    print(f"Spatial predictions array shape: {spatial_array.shape}")
    
    # Apply ensemble method
    if ensemble_method == "average":
        # Simple averaging across models
        temporal_ensemble = np.mean(temporal_array, axis=0)  # Shape: (2, 5, n_samples)
        spatial_ensemble = np.mean(spatial_array, axis=0)
        
        print(f"\nEnsemble shapes after averaging:")
        print(f"  Temporal: {temporal_ensemble.shape}")
        print(f"  Spatial: {spatial_ensemble.shape}")
        
    elif ensemble_method == "weighted_average":
        # TODO: Implement weighted averaging based on model performance
        raise NotImplementedError("Weighted averaging not yet implemented")
    
    elif ensemble_method == "voting":
        # TODO: Implement voting mechanism
        raise NotImplementedError("Voting not yet implemented")
    
    else:
        raise ValueError(f"Unknown ensemble method: {ensemble_method}")
    
    # Generate timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save ensemble predictions
    temporal_path = os.path.join(output_dir, f"ensemble_temporal_predictions_{timestamp}.h5")
    spatial_path = os.path.join(output_dir, f"ensemble_spatial_predictions_{timestamp}.h5")
    
    # Save temporal predictions
    save_ensemble_predictions(temporal_ensemble, temporal_path, "temporal")
    
    # Save spatial predictions
    save_ensemble_predictions(spatial_ensemble, spatial_path, "spatial")
    
    # Create metadata
    ensemble_metadata = {
        "timestamp": timestamp,
        "ensemble_method": ensemble_method,
        "num_models": len(model_identifiers),
        "model_identifiers": model_identifiers,
        "temporal_shape": temporal_ensemble.shape,
        "spatial_shape": spatial_ensemble.shape,
        "temporal_path": temporal_path,
        "spatial_path": spatial_path
    }
    
    # Save metadata
    metadata_path = os.path.join(output_dir, f"ensemble_metadata_{timestamp}.json")
    save_ensemble_metadata(ensemble_metadata, metadata_path)
    
    print(f"\n=== Ensemble Predictions Saved ===")
    print(f"Temporal: {temporal_path}")
    print(f"Spatial: {spatial_path}")
    print(f"Metadata: {metadata_path}")
    
    return temporal_path, spatial_path, ensemble_metadata


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
        
        # Save as CIF for compatibility
        f.create_dataset('cif', data=predictions, compression='gzip')
        
        # Add metadata attributes
        f.attrs['dataset_type'] = dataset_type
        f.attrs['shape'] = predictions.shape
        f.attrs['description'] = f"Ensemble {dataset_type} predictions"
        f.attrs['format'] = "CIF (Cumulative Incidence Function)"
        f.attrs['dimensions'] = "events x time_points x samples"
        
        # Add time points information
        time_points = np.array([365, 730, 1095, 1460, 1825])
        f.create_dataset('time_points', data=time_points)
        
        # Add event labels
        event_labels = np.array(['Event 1', 'Event 2'])
        f.create_dataset('event_labels', data=event_labels.astype('S'))
        
    print(f"Saved {dataset_type} ensemble predictions to {file_path}")


def save_ensemble_metadata(metadata: Dict[str, Any], file_path: str) -> None:
    """
    Save ensemble metadata to JSON file.
    
    Args:
        metadata: Metadata dictionary
        file_path: Path to save the file
    """
    import json
    
    # Convert numpy arrays to lists for JSON serialization
    metadata_serializable = {}
    for key, value in metadata.items():
        if isinstance(value, np.ndarray):
            metadata_serializable[key] = value.tolist()
        elif isinstance(value, (np.int64, np.int32)):
            metadata_serializable[key] = int(value)
        elif isinstance(value, (np.float64, np.float32)):
            metadata_serializable[key] = float(value)
        elif isinstance(value, tuple):
            metadata_serializable[key] = list(value)
        else:
            metadata_serializable[key] = value
    
    with open(file_path, 'w') as f:
        json.dump(metadata_serializable, f, indent=2)
    
    print(f"Saved ensemble metadata to {file_path}")


def calculate_ensemble_statistics(predictions_array: np.ndarray) -> Dict[str, Any]:
    """
    Calculate statistics for the ensemble predictions.
    
    Args:
        predictions_array: Array of predictions from multiple models
        
    Returns:
        Dictionary of statistics
    """
    stats = {
        "mean": np.mean(predictions_array, axis=0),
        "std": np.std(predictions_array, axis=0),
        "min": np.min(predictions_array, axis=0),
        "max": np.max(predictions_array, axis=0),
        "median": np.median(predictions_array, axis=0)
    }
    
    return stats