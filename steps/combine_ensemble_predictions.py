"""
Step to combine DeepSurv and DeepHit predictions into final ensemble.
"""

import numpy as np
from typing import Dict, Tuple, Annotated
from zenml import step


@step
def combine_ensemble_predictions(
    deepsurv_temporal_preds: Dict[str, np.ndarray],
    deepsurv_spatial_preds: Dict[str, np.ndarray],
    deephit_temporal_preds: Dict[str, np.ndarray],
    deephit_spatial_preds: Dict[str, np.ndarray]
) -> Tuple[
    Annotated[np.ndarray, "ensemble_temporal_predictions"],
    Annotated[np.ndarray, "ensemble_spatial_predictions"]
]:
    """
    Combine all predictions using simple averaging.
    
    Args:
        deepsurv_temporal_preds: Dictionary of DeepSurv temporal predictions
        deepsurv_spatial_preds: Dictionary of DeepSurv spatial predictions
        deephit_temporal_preds: Dictionary of DeepHit temporal predictions
        deephit_spatial_preds: Dictionary of DeepHit spatial predictions
        
    Returns:
        Tuple of ensemble predictions (temporal, spatial) with shape (2, 5, n_samples)
    """
    print("\nCombining ensemble predictions...")
    
    # Collect all temporal predictions
    all_temporal_preds = []
    
    # Add DeepSurv predictions
    for group_id, preds in deepsurv_temporal_preds.items():
        all_temporal_preds.append(preds)
        print(f"  Added DeepSurv {group_id} temporal: {preds.shape}")
    
    # Add DeepHit predictions
    for model_name, preds in deephit_temporal_preds.items():
        all_temporal_preds.append(preds)
        print(f"  Added {model_name} temporal: {preds.shape}")
    
    # Stack and average temporal predictions
    temporal_stack = np.stack(all_temporal_preds, axis=0)  # Shape: (24, 2, 5, n_samples)
    ensemble_temporal = np.mean(temporal_stack, axis=0)    # Shape: (2, 5, n_samples)
    
    print(f"\nTemporal ensemble shape: {ensemble_temporal.shape}")
    print(f"  Combined {len(all_temporal_preds)} models")
    
    # Collect all spatial predictions
    all_spatial_preds = []
    
    # Add DeepSurv predictions
    for group_id, preds in deepsurv_spatial_preds.items():
        all_spatial_preds.append(preds)
        print(f"  Added DeepSurv {group_id} spatial: {preds.shape}")
    
    # Add DeepHit predictions
    for model_name, preds in deephit_spatial_preds.items():
        all_spatial_preds.append(preds)
        print(f"  Added {model_name} spatial: {preds.shape}")
    
    # Stack and average spatial predictions
    spatial_stack = np.stack(all_spatial_preds, axis=0)  # Shape: (24, 2, 5, n_samples)
    ensemble_spatial = np.mean(spatial_stack, axis=0)    # Shape: (2, 5, n_samples)
    
    print(f"\nSpatial ensemble shape: {ensemble_spatial.shape}")
    print(f"  Combined {len(all_spatial_preds)} models")
    
    # Verify shapes
    assert ensemble_temporal.shape[0] == 2, f"Expected 2 events, got {ensemble_temporal.shape[0]}"
    assert ensemble_temporal.shape[1] == 5, f"Expected 5 time points, got {ensemble_temporal.shape[1]}"
    assert ensemble_spatial.shape[0] == 2, f"Expected 2 events, got {ensemble_spatial.shape[0]}"
    assert ensemble_spatial.shape[1] == 5, f"Expected 5 time points, got {ensemble_spatial.shape[1]}"
    
    print("\nEnsemble combination complete!")
    
    return ensemble_temporal, ensemble_spatial