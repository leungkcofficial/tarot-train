"""
Step to evaluate the final ensemble predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Annotated
from zenml import step
from src.evaluation_metrics import calculate_all_metrics, concordance_index_censored


@step
def evaluate_final_ensemble(
    ensemble_temporal_preds: np.ndarray,
    ensemble_spatial_preds: np.ndarray
) -> Annotated[Dict, "evaluation_results"]:
    """
    Evaluate the final ensemble predictions.
    
    Args:
        ensemble_temporal_preds: Ensemble temporal predictions (2, 5, n_samples)
        ensemble_spatial_preds: Ensemble spatial predictions (2, 5, n_samples)
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Load test labels from saved files
    import os
    import pickle
    
    # Load temporal test labels
    temporal_test_path = "data/processed/temporal_test.pkl"
    if os.path.exists(temporal_test_path):
        with open(temporal_test_path, 'rb') as f:
            temporal_test_data = pickle.load(f)
            y_temporal_test = temporal_test_data['y']
    else:
        # Try alternative path
        temporal_test_path = "results/final_deploy/temporal_test_labels.pkl"
        if os.path.exists(temporal_test_path):
            with open(temporal_test_path, 'rb') as f:
                y_temporal_test = pickle.load(f)
        else:
            raise FileNotFoundError(f"Could not find temporal test labels at {temporal_test_path}")
    
    # Load spatial test labels
    spatial_test_path = "data/processed/spatial_test.pkl"
    if os.path.exists(spatial_test_path):
        with open(spatial_test_path, 'rb') as f:
            spatial_test_data = pickle.load(f)
            y_spatial_test = spatial_test_data['y']
    else:
        # Try alternative path
        spatial_test_path = "results/final_deploy/spatial_test_labels.pkl"
        if os.path.exists(spatial_test_path):
            with open(spatial_test_path, 'rb') as f:
                y_spatial_test = pickle.load(f)
        else:
            raise FileNotFoundError(f"Could not find spatial test labels at {spatial_test_path}")
    print("\nEvaluating final ensemble...")
    
    # Time points for evaluation (in days)
    time_points = np.array([365, 730, 1095, 1460, 1825])
    
    results = {
        "temporal": {},
        "spatial": {},
        "summary": {}
    }
    
    # Evaluate temporal predictions
    print("\nEvaluating temporal predictions...")
    
    # Get times and events
    temporal_times = y_temporal_test['time'].values
    temporal_events = y_temporal_test['event'].values
    
    # Calculate metrics for the full competing risks model
    temporal_metrics = calculate_all_metrics(
        times=temporal_times,
        events=temporal_events,
        predictions=ensemble_temporal_preds,
        time_points=time_points
    )
    
    results["temporal"]["combined"] = temporal_metrics
    print(f"  Combined IBS: {temporal_metrics.get('ibs', 'N/A'):.4f}")
    print(f"  Event 1 C-index: {temporal_metrics.get('cidx_event1', 'N/A'):.4f}")
    print(f"  Event 2 C-index: {temporal_metrics.get('cidx_event2', 'N/A'):.4f}")
    print(f"  Event 1 IBS: {temporal_metrics.get('ibs_event1', 'N/A'):.4f}")
    print(f"  Event 2 IBS: {temporal_metrics.get('ibs_event2', 'N/A'):.4f}")
    
    # Store individual event metrics for compatibility
    results["temporal"]["Event_1"] = {
        "c_index": temporal_metrics.get('cidx_event1', np.nan),
        "ibs": temporal_metrics.get('ibs_event1', np.nan)
    }
    results["temporal"]["Event_2"] = {
        "c_index": temporal_metrics.get('cidx_event2', np.nan),
        "ibs": temporal_metrics.get('ibs_event2', np.nan)
    }
    
    # Evaluate spatial predictions
    print("\nEvaluating spatial predictions...")
    
    # Get times and events
    spatial_times = y_spatial_test['time'].values
    spatial_events = y_spatial_test['event'].values
    
    # Calculate metrics for the full competing risks model
    spatial_metrics = calculate_all_metrics(
        times=spatial_times,
        events=spatial_events,
        predictions=ensemble_spatial_preds,
        time_points=time_points
    )
    
    results["spatial"]["combined"] = spatial_metrics
    print(f"  Combined IBS: {spatial_metrics.get('ibs', 'N/A'):.4f}")
    print(f"  Event 1 C-index: {spatial_metrics.get('cidx_event1', 'N/A'):.4f}")
    print(f"  Event 2 C-index: {spatial_metrics.get('cidx_event2', 'N/A'):.4f}")
    print(f"  Event 1 IBS: {spatial_metrics.get('ibs_event1', 'N/A'):.4f}")
    print(f"  Event 2 IBS: {spatial_metrics.get('ibs_event2', 'N/A'):.4f}")
    
    # Store individual event metrics for compatibility
    results["spatial"]["Event_1"] = {
        "c_index": spatial_metrics.get('cidx_event1', np.nan),
        "ibs": spatial_metrics.get('ibs_event2', np.nan)
    }
    results["spatial"]["Event_2"] = {
        "c_index": spatial_metrics.get('cidx_event2', np.nan),
        "ibs": spatial_metrics.get('ibs_event2', np.nan)
    }
    
    # Calculate summary statistics
    results["summary"] = {
        "temporal_avg_c_index": np.nanmean([
            results["temporal"]["Event_1"].get("c_index", np.nan),
            results["temporal"]["Event_2"].get("c_index", np.nan)
        ]),
        "spatial_avg_c_index": np.nanmean([
            results["spatial"]["Event_1"].get("c_index", np.nan),
            results["spatial"]["Event_2"].get("c_index", np.nan)
        ]),
        "temporal_avg_ibs": np.nanmean([
            results["temporal"]["Event_1"].get("ibs", np.nan),
            results["temporal"]["Event_2"].get("ibs", np.nan)
        ]),
        "spatial_avg_ibs": np.nanmean([
            results["spatial"]["Event_1"].get("ibs", np.nan),
            results["spatial"]["Event_2"].get("ibs", np.nan)
        ]),
        "temporal_combined_ibs": results["temporal"]["combined"].get("ibs", np.nan),
        "spatial_combined_ibs": results["spatial"]["combined"].get("ibs", np.nan)
    }
    
    print("\n" + "="*60)
    print("ENSEMBLE EVALUATION SUMMARY")
    print("="*60)
    print(f"Temporal Average C-index: {results['summary']['temporal_avg_c_index']:.4f}")
    print(f"Spatial Average C-index: {results['summary']['spatial_avg_c_index']:.4f}")
    print(f"Temporal Average IBS: {results['summary']['temporal_avg_ibs']:.4f}")
    print(f"Spatial Average IBS: {results['summary']['spatial_avg_ibs']:.4f}")
    
    return results