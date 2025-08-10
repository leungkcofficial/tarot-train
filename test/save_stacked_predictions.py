"""
Script to save the stacked CIF predictions for both temporal and spatial test sets.
This creates the (24, 2, 5, n_samples) arrays that can be reused.

Order according to model_grouping_summary.md:
- [0-11]: Groups 1-12 (DeepSurv models paired by event)
- [12-23]: Models 25-36 (DeepHit models)
"""

import numpy as np
import pickle
import h5py
import os
from steps.evaluate_ensemble_combinations_dataframe import stack_deepsurv_models


def verify_stacked_shape_and_order(stacked_array, dataset_name):
    """Verify the shape and print the order of models in the stacked array."""
    
    print(f"\n{'='*60}")
    print(f"Verifying {dataset_name} stacked predictions")
    print(f"{'='*60}")
    
    # Check shape
    expected_dims = 4
    if len(stacked_array.shape) != expected_dims:
        raise ValueError(f"Expected 4 dimensions, got {len(stacked_array.shape)}")
    
    n_models, n_events, n_times, n_samples = stacked_array.shape
    
    print(f"Shape: {stacked_array.shape}")
    print(f"  - Number of model groups: {n_models}")
    print(f"  - Number of events: {n_events}")
    print(f"  - Number of time points: {n_times}")
    print(f"  - Number of samples: {n_samples}")
    
    # Verify expected values
    assert n_models == 24, f"Expected 24 model groups, got {n_models}"
    assert n_events == 2, f"Expected 2 events, got {n_events}"
    assert n_times == 5, f"Expected 5 time points, got {n_times}"
    
    # Print model order
    print(f"\nModel order in stacked array:")
    print("DeepSurv Groups (indices 0-11):")
    group_mapping = {
        0: "Group 1: Models 1 (Event 1) + 2 (Event 2)",
        1: "Group 2: Models 3 (Event 1) + 4 (Event 2)",
        2: "Group 3: Models 5 (Event 1) + 6 (Event 2)",
        3: "Group 4: Models 7 (Event 1) + 8 (Event 2)",
        4: "Group 5: Models 9 (Event 1) + 10 (Event 2)",
        5: "Group 6: Models 11 (Event 1) + 12 (Event 2)",
        6: "Group 7: Models 13 (Event 1) + 14 (Event 2)",
        7: "Group 8: Models 15 (Event 1) + 16 (Event 2)",
        8: "Group 9: Models 17 (Event 1) + 18 (Event 2)",
        9: "Group 10: Models 19 (Event 1) + 20 (Event 2)",
        10: "Group 11: Models 21 (Event 1) + 22 (Event 2)",
        11: "Group 12: Models 23 (Event 1) + 24 (Event 2)"
    }
    
    for idx, desc in group_mapping.items():
        print(f"  [{idx}] {desc}")
    
    print("\nDeepHit Models (indices 12-23):")
    for i in range(12):
        model_id = 25 + i
        print(f"  [{12 + i}] Model {model_id}")
    
    print(f"\n✓ Shape verification passed for {dataset_name}")
    return True


def save_stacked_predictions():
    """Load individual predictions and save stacked versions."""
    
    print("Loading individual predictions...")
    
    # Load predictions
    with open('results/final_deploy/temporal_predictions.pkl', 'rb') as f:
        temporal_predictions = pickle.load(f)
    
    with open('results/final_deploy/spatial_predictions.pkl', 'rb') as f:
        spatial_predictions = pickle.load(f)
    
    print(f"Loaded {len(temporal_predictions)} temporal predictions")
    print(f"Loaded {len(spatial_predictions)} spatial predictions")
    
    # Stack predictions
    print("\nStacking predictions according to model_grouping_summary.md...")
    temporal_stacked = stack_deepsurv_models(temporal_predictions)
    spatial_stacked = stack_deepsurv_models(spatial_predictions)
    
    # Verify shapes and order
    verify_stacked_shape_and_order(temporal_stacked, "Temporal")
    verify_stacked_shape_and_order(spatial_stacked, "Spatial")
    
    # Save stacked predictions
    output_dir = "results/final_deploy/stacked_predictions"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as H5 files for efficient storage
    print(f"\nSaving stacked predictions to {output_dir}...")
    
    # Save temporal stacked predictions
    temporal_file = os.path.join(output_dir, "temporal_stacked_cif.h5")
    with h5py.File(temporal_file, 'w') as f:
        f.create_dataset('stacked_cif', data=temporal_stacked, compression='gzip')
        f.attrs['shape'] = temporal_stacked.shape
        f.attrs['description'] = 'Stacked temporal CIF predictions (24 models, 2 events, 5 time points, n_samples)'
        f.attrs['time_points'] = [365, 730, 1095, 1460, 1825]
    print(f"Saved: {temporal_file}")
    
    # Save spatial stacked predictions
    spatial_file = os.path.join(output_dir, "spatial_stacked_cif.h5")
    with h5py.File(spatial_file, 'w') as f:
        f.create_dataset('stacked_cif', data=spatial_stacked, compression='gzip')
        f.attrs['shape'] = spatial_stacked.shape
        f.attrs['description'] = 'Stacked spatial CIF predictions (24 models, 2 events, 5 time points, n_samples)'
        f.attrs['time_points'] = [365, 730, 1095, 1460, 1825]
    print(f"Saved: {spatial_file}")
    
    # Also save as pickle for convenience
    temporal_pkl = os.path.join(output_dir, "temporal_stacked_cif.pkl")
    with open(temporal_pkl, 'wb') as f:
        pickle.dump(temporal_stacked, f)
    print(f"Saved: {temporal_pkl}")
    
    spatial_pkl = os.path.join(output_dir, "spatial_stacked_cif.pkl")
    with open(spatial_pkl, 'wb') as f:
        pickle.dump(spatial_stacked, f)
    print(f"Saved: {spatial_pkl}")
    
    # Save metadata with detailed model mapping
    metadata = {
        'temporal_shape': temporal_stacked.shape,
        'spatial_shape': spatial_stacked.shape,
        'time_points': [365, 730, 1095, 1460, 1825],
        'n_models': 24,
        'n_events': 2,
        'model_order': {
            'indices_0_11': 'DeepSurv Groups 1-12 (paired by event)',
            'indices_12_23': 'DeepHit Models 25-36',
            'detailed_mapping': {
                0: 'Group 1: Models 1+2',
                1: 'Group 2: Models 3+4',
                2: 'Group 3: Models 5+6',
                3: 'Group 4: Models 7+8',
                4: 'Group 5: Models 9+10',
                5: 'Group 6: Models 11+12',
                6: 'Group 7: Models 13+14',
                7: 'Group 8: Models 15+16',
                8: 'Group 9: Models 17+18',
                9: 'Group 10: Models 19+20',
                10: 'Group 11: Models 21+22',
                11: 'Group 12: Models 23+24',
                12: 'Model 25', 13: 'Model 26', 14: 'Model 27', 15: 'Model 28',
                16: 'Model 29', 17: 'Model 30', 18: 'Model 31', 19: 'Model 32',
                20: 'Model 33', 21: 'Model 34', 22: 'Model 35', 23: 'Model 36'
            }
        },
        'description': 'Stacked CIF predictions with DeepSurv models grouped by event pairs and DeepHit models as-is'
    }
    
    metadata_file = os.path.join(output_dir, "stacking_metadata.pkl")
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Saved: {metadata_file}")
    
    print("\n✓ Stacked predictions saved successfully!")
    print(f"\nYou can load them using:")
    print("```python")
    print("import h5py")
    print("with h5py.File('results/final_deploy/stacked_predictions/temporal_stacked_cif.h5', 'r') as f:")
    print("    temporal_stacked = f['stacked_cif'][:]")
    print("```")


if __name__ == "__main__":
    save_stacked_predictions()