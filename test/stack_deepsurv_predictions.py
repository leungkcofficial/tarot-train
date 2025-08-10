"""
Stack DeepSurv predictions according to model_grouping_summary.md

This script:
1. Loads DeepSurv predictions from individual H5 files
2. Extracts predictions at 5 specific time points
3. Stacks Event 1 and Event 2 predictions for each group
4. Saves stacked predictions for use in ensemble pipeline
"""

import os
import json
import h5py
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple


# Define the 12 DeepSurv groups based on model_grouping_summary.md
DEEPSURV_GROUPS = {
    "group_1": {
        "name": "ANN_None_CI",
        "event1": "Ensemble_model1_DeepSurv_ANN_Event_1",
        "event2": "Ensemble_model3_DeepSurv_ANN_Event_2"
    },
    "group_2": {
        "name": "ANN_None_LL",
        "event1": "Ensemble_model2_DeepSurv_ANN_Event_1",
        "event2": "Ensemble_model4_DeepSurv_ANN_Event_2"
    },
    "group_3": {
        "name": "ANN_NearMiss1_CI",
        "event1": "Ensemble_model5_DeepSurv_ANN_Event_1",
        "event2": "Ensemble_model7_DeepSurv_ANN_Event_2"
    },
    "group_4": {
        "name": "ANN_NearMiss1_LL",
        "event1": "Ensemble_model6_DeepSurv_ANN_Event_1",
        "event2": "Ensemble_model8_DeepSurv_ANN_Event_2"
    },
    "group_5": {
        "name": "ANN_KNN_CI",
        "event1": "Ensemble_model9_DeepSurv_ANN_Event_1",
        "event2": "Ensemble_model11_DeepSurv_ANN_Event_2"
    },
    "group_6": {
        "name": "ANN_KNN_LL",
        "event1": "Ensemble_model10_DeepSurv_ANN_Event_1",
        "event2": "Ensemble_model12_DeepSurv_ANN_Event_2"
    },
    "group_7": {
        "name": "LSTM_None_CI",
        "event1": "Ensemble_model13_DeepSurv_LSTM_Event_1",
        "event2": "Ensemble_model15_DeepSurv_LSTM_Event_2"
    },
    "group_8": {
        "name": "LSTM_None_LL",
        "event1": "Ensemble_model14_DeepSurv_LSTM_Event_1",
        "event2": "Ensemble_model16_DeepSurv_LSTM_Event_2"
    },
    "group_9": {
        "name": "LSTM_NearMiss3_CI",
        "event1": "Ensemble_model17_DeepSurv_LSTM_Event_1",
        "event2": "Ensemble_model19_DeepSurv_LSTM_Event_2"
    },
    "group_10": {
        "name": "LSTM_NearMiss3_LL",
        "event1": "Ensemble_model18_DeepSurv_LSTM_Event_1",
        "event2": "Ensemble_model20_DeepSurv_LSTM_Event_2"
    },
    "group_11": {
        "name": "LSTM_KNN_CI",
        "event1": "Ensemble_model21_DeepSurv_LSTM_Event_1",
        "event2": "Ensemble_model23_DeepSurv_LSTM_Event_2"
    },
    "group_12": {
        "name": "LSTM_KNN_LL",
        "event1": "Ensemble_model22_DeepSurv_LSTM_Event_1",
        "event2": "Ensemble_model24_DeepSurv_LSTM_Event_2"
    }
}

# Time points to extract (in days)
TIME_POINTS = [365, 730, 1095, 1460, 1825]


def load_predictions(h5_path: str) -> np.ndarray:
    """Load predictions from H5 file."""
    with h5py.File(h5_path, 'r') as f:
        return f['predictions'][:]


def extract_at_time_points(predictions: np.ndarray, time_points: List[int]) -> np.ndarray:
    """
    Extract predictions at specific time points.
    
    Args:
        predictions: Shape (n_times, n_samples) for DeepSurv
        time_points: List of time points (days) to extract
        
    Returns:
        Array of shape (n_time_points, n_samples)
    """
    # DeepSurv predictions are indexed by day (0-based)
    # So day 365 is at index 364
    indices = [t - 1 for t in time_points]
    return predictions[indices, :]


def stack_deepsurv_group(event1_pred: np.ndarray, event2_pred: np.ndarray) -> np.ndarray:
    """
    Stack Event 1 and Event 2 predictions.
    
    Args:
        event1_pred: Shape (n_time_points, n_samples)
        event2_pred: Shape (n_time_points, n_samples)
        
    Returns:
        Stacked array of shape (2, n_time_points, n_samples)
    """
    return np.stack([event1_pred, event2_pred], axis=0)


def main():
    """Main function to stack DeepSurv predictions."""
    # Paths
    predictions_dir = "results/final_deploy/individual_predictions"
    output_dir = "results/final_deploy/stacked_predictions"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load prediction summary to get file paths
    summary_path = os.path.join(predictions_dir, "prediction_generation_summary_20250808_113457.json")
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    # Create a mapping from model name to info
    model_map = {info["model_name"]: info for info in summary["results"]}
    
    # Results storage
    results = {
        "temporal": {},
        "spatial": {},
        "metadata": {
            "time_points": TIME_POINTS,
            "n_groups": len(DEEPSURV_GROUPS),
            "timestamp": datetime.now().isoformat()
        }
    }
    
    # Process each group
    for group_id, group_info in DEEPSURV_GROUPS.items():
        print(f"\nProcessing {group_id}: {group_info['name']}")
        
        # Get model info
        event1_info = model_map[group_info["event1"]]
        event2_info = model_map[group_info["event2"]]
        
        # Process temporal predictions
        print("  - Loading temporal predictions...")
        event1_temporal = load_predictions(event1_info["temporal_predictions_path"])
        event2_temporal = load_predictions(event2_info["temporal_predictions_path"])
        
        # Extract at time points
        event1_temporal_extracted = extract_at_time_points(event1_temporal, TIME_POINTS)
        event2_temporal_extracted = extract_at_time_points(event2_temporal, TIME_POINTS)
        
        # Stack
        temporal_stacked = stack_deepsurv_group(event1_temporal_extracted, event2_temporal_extracted)
        
        # Save temporal
        temporal_path = os.path.join(output_dir, f"{group_id}_temporal_stacked.h5")
        with h5py.File(temporal_path, 'w') as f:
            f.create_dataset('predictions', data=temporal_stacked)
            f.attrs['group_name'] = group_info['name']
            f.attrs['event1_model'] = group_info['event1']
            f.attrs['event2_model'] = group_info['event2']
            f.attrs['time_points'] = TIME_POINTS
            f.attrs['shape'] = temporal_stacked.shape
        
        print(f"    Saved temporal: {temporal_stacked.shape}")
        
        # Process spatial predictions
        print("  - Loading spatial predictions...")
        event1_spatial = load_predictions(event1_info["spatial_predictions_path"])
        event2_spatial = load_predictions(event2_info["spatial_predictions_path"])
        
        # Extract at time points
        event1_spatial_extracted = extract_at_time_points(event1_spatial, TIME_POINTS)
        event2_spatial_extracted = extract_at_time_points(event2_spatial, TIME_POINTS)
        
        # Stack
        spatial_stacked = stack_deepsurv_group(event1_spatial_extracted, event2_spatial_extracted)
        
        # Save spatial
        spatial_path = os.path.join(output_dir, f"{group_id}_spatial_stacked.h5")
        with h5py.File(spatial_path, 'w') as f:
            f.create_dataset('predictions', data=spatial_stacked)
            f.attrs['group_name'] = group_info['name']
            f.attrs['event1_model'] = group_info['event1']
            f.attrs['event2_model'] = group_info['event2']
            f.attrs['time_points'] = TIME_POINTS
            f.attrs['shape'] = spatial_stacked.shape
        
        print(f"    Saved spatial: {spatial_stacked.shape}")
        
        # Store results
        results["temporal"][group_id] = {
            "path": temporal_path,
            "shape": list(temporal_stacked.shape),
            "group_name": group_info['name'],
            "event1_model": group_info['event1'],
            "event2_model": group_info['event2']
        }
        
        results["spatial"][group_id] = {
            "path": spatial_path,
            "shape": list(spatial_stacked.shape),
            "group_name": group_info['name'],
            "event1_model": group_info['event1'],
            "event2_model": group_info['event2']
        }
    
    # Save summary
    summary_path = os.path.join(output_dir, "stacking_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("STACKING COMPLETE")
    print(f"{'='*60}")
    print(f"Processed {len(DEEPSURV_GROUPS)} groups")
    print(f"Time points extracted: {TIME_POINTS}")
    print(f"Output directory: {output_dir}")
    print(f"Summary saved to: {summary_path}")
    
    # Print shape summary
    print("\nShape Summary:")
    for group_id in DEEPSURV_GROUPS:
        temporal_shape = results["temporal"][group_id]["shape"]
        spatial_shape = results["spatial"][group_id]["shape"]
        print(f"  {group_id}: temporal={temporal_shape}, spatial={spatial_shape}")


if __name__ == "__main__":
    main()