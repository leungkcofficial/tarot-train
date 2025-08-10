"""
Ensemble Evaluator Step for Ensemble Deployment

This module contains the ZenML step for evaluating ensemble predictions performance.
"""

import os
import json
import numpy as np
import pandas as pd
import h5py
from datetime import datetime
from zenml.steps import step
from typing import Dict, Any, Tuple


@step(enable_cache=True)
def ensemble_evaluator(
    temporal_ensemble_path: str,
    spatial_ensemble_path: str,
    ensemble_metadata: Dict[str, Any],
    temporal_test_df: pd.DataFrame,
    spatial_test_df: pd.DataFrame,
    master_df_mapping_path: str = "src/default_master_df_mapping.yml",
    output_dir: str = "results/final_deploy/ensemble_eval"
) -> Dict[str, Any]:
    """
    Evaluate the performance of ensemble predictions.
    
    Args:
        temporal_ensemble_path: Path to temporal ensemble predictions
        spatial_ensemble_path: Path to spatial ensemble predictions
        ensemble_metadata: Metadata from ensemble_predictions step
        temporal_test_df: Original temporal test dataframe (with labels)
        spatial_test_df: Original spatial test dataframe (with labels)
        master_df_mapping_path: Path to master dataframe mapping
        output_dir: Directory to save evaluation results
        
    Returns:
        Dictionary containing evaluation metrics
    """
    from src.util import load_yaml_file
    from src.competing_risks_evaluation import evaluate_competing_risks_predictions
    
    print("\n=== Evaluating Ensemble Predictions ===")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load master dataframe mapping
    master_df_mapping = load_yaml_file(master_df_mapping_path)
    duration_col = master_df_mapping.get("duration", "duration")
    event_col = master_df_mapping.get("endpoint", "endpoint")
    
    # Load ensemble predictions
    print("\nLoading ensemble predictions...")
    temporal_predictions = load_predictions_from_h5(temporal_ensemble_path)
    spatial_predictions = load_predictions_from_h5(spatial_ensemble_path)
    
    print(f"Temporal predictions shape: {temporal_predictions.shape}")
    print(f"Spatial predictions shape: {spatial_predictions.shape}")
    
    # Prepare evaluation data
    evaluation_results = {}
    
    # Evaluate temporal predictions
    print("\n--- Evaluating Temporal Test Set ---")
    temporal_metrics = evaluate_dataset(
        predictions=temporal_predictions,
        test_df=temporal_test_df,
        duration_col=duration_col,
        event_col=event_col,
        dataset_name="temporal"
    )
    evaluation_results['temporal'] = temporal_metrics
    
    # Evaluate spatial predictions
    print("\n--- Evaluating Spatial Test Set ---")
    spatial_metrics = evaluate_dataset(
        predictions=spatial_predictions,
        test_df=spatial_test_df,
        duration_col=duration_col,
        event_col=event_col,
        dataset_name="spatial"
    )
    evaluation_results['spatial'] = spatial_metrics
    
    # Calculate combined metrics
    print("\n--- Combined Metrics ---")
    combined_metrics = calculate_combined_metrics(temporal_metrics, spatial_metrics)
    evaluation_results['combined'] = combined_metrics
    
    # Add metadata
    evaluation_results['metadata'] = {
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'ensemble_method': ensemble_metadata.get('ensemble_method'),
        'num_models': ensemble_metadata.get('num_models'),
        'temporal_samples': len(temporal_test_df),
        'spatial_samples': len(spatial_test_df)
    }
    
    # Save evaluation results
    timestamp = evaluation_results['metadata']['timestamp']
    results_path = os.path.join(output_dir, f"ensemble_evaluation_{timestamp}.json")
    save_evaluation_results(evaluation_results, results_path)
    
    # Print summary
    print_evaluation_summary(evaluation_results)
    
    return evaluation_results


def evaluate_dataset(
    predictions: np.ndarray,
    test_df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    dataset_name: str
) -> Dict[str, Any]:
    """
    Evaluate predictions on a single dataset.
    
    Args:
        predictions: Predictions array (2, 5, n_samples)
        test_df: Test dataframe with true labels
        duration_col: Name of duration column
        event_col: Name of event column
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary of evaluation metrics
    """
    from src.competing_risks_evaluation import (
        calculate_concordance_index,
        calculate_brier_score,
        calculate_integrated_brier_score,
        calculate_cumulative_dynamic_auc
    )
    
    # Extract true labels
    durations = test_df[duration_col].values
    events = test_df[event_col].values
    
    # Time points for evaluation
    time_points = np.array([365, 730, 1095, 1460, 1825])
    
    metrics = {
        'dataset': dataset_name,
        'n_samples': len(test_df),
        'event_distribution': pd.Series(events).value_counts().to_dict()
    }
    
    # Evaluate each event type
    for event_idx in range(2):
        event_name = f"Event_{event_idx + 1}"
        event_predictions = predictions[event_idx, :, :]  # Shape: (5, n_samples)
        
        print(f"\n  Evaluating {event_name}:")
        
        # C-index (using predictions at last time point)
        try:
            c_index = calculate_concordance_index(
                durations=durations,
                events=(events == event_idx + 1).astype(int),
                predictions=event_predictions[-1, :],  # Last time point
                event_type=event_idx + 1
            )
            print(f"    C-index: {c_index:.4f}")
        except Exception as e:
            print(f"    C-index calculation failed: {e}")
            c_index = None
        
        # Brier scores at each time point
        brier_scores = []
        for t_idx, t in enumerate(time_points):
            try:
                bs = calculate_brier_score(
                    durations=durations,
                    events=events,
                    predictions=event_predictions[t_idx, :],
                    time_point=t,
                    event_type=event_idx + 1
                )
                brier_scores.append(bs)
            except Exception as e:
                print(f"    Brier score at t={t} failed: {e}")
                brier_scores.append(None)
        
        # Integrated Brier Score
        try:
            ibs = calculate_integrated_brier_score(
                durations=durations,
                events=events,
                predictions=event_predictions,
                time_points=time_points,
                event_type=event_idx + 1
            )
            print(f"    Integrated Brier Score: {ibs:.4f}")
        except Exception as e:
            print(f"    IBS calculation failed: {e}")
            ibs = None
        
        # Store metrics
        metrics[event_name] = {
            'c_index': c_index,
            'brier_scores': brier_scores,
            'integrated_brier_score': ibs,
            'time_points': time_points.tolist()
        }
    
    return metrics


def calculate_combined_metrics(temporal_metrics: Dict, spatial_metrics: Dict) -> Dict[str, Any]:
    """
    Calculate combined metrics across temporal and spatial datasets.
    
    Args:
        temporal_metrics: Metrics from temporal dataset
        spatial_metrics: Metrics from spatial dataset
        
    Returns:
        Combined metrics
    """
    combined = {}
    
    # Average C-indices
    for event in ['Event_1', 'Event_2']:
        temporal_c = temporal_metrics.get(event, {}).get('c_index')
        spatial_c = spatial_metrics.get(event, {}).get('c_index')
        
        if temporal_c is not None and spatial_c is not None:
            combined[f'{event}_avg_c_index'] = (temporal_c + spatial_c) / 2
            combined[f'{event}_temporal_c_index'] = temporal_c
            combined[f'{event}_spatial_c_index'] = spatial_c
    
    # Average IBS
    for event in ['Event_1', 'Event_2']:
        temporal_ibs = temporal_metrics.get(event, {}).get('integrated_brier_score')
        spatial_ibs = spatial_metrics.get(event, {}).get('integrated_brier_score')
        
        if temporal_ibs is not None and spatial_ibs is not None:
            combined[f'{event}_avg_ibs'] = (temporal_ibs + spatial_ibs) / 2
            combined[f'{event}_temporal_ibs'] = temporal_ibs
            combined[f'{event}_spatial_ibs'] = spatial_ibs
    
    return combined


def load_predictions_from_h5(file_path: str) -> np.ndarray:
    """
    Load predictions from HDF5 file.
    
    Args:
        file_path: Path to HDF5 file
        
    Returns:
        Predictions array
    """
    with h5py.File(file_path, 'r') as f:
        if 'predictions' in f:
            predictions = f['predictions'][:]
        elif 'cif' in f:
            predictions = f['cif'][:]
        else:
            raise ValueError(f"No predictions found in {file_path}")
    
    return predictions


def save_evaluation_results(results: Dict[str, Any], file_path: str) -> None:
    """
    Save evaluation results to JSON file.
    
    Args:
        results: Evaluation results dictionary
        file_path: Path to save the file
    """
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(file_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nSaved evaluation results to {file_path}")


def print_evaluation_summary(results: Dict[str, Any]) -> None:
    """
    Print a summary of evaluation results.
    
    Args:
        results: Evaluation results dictionary
    """
    print("\n" + "="*60)
    print("ENSEMBLE EVALUATION SUMMARY")
    print("="*60)
    
    # Combined metrics
    combined = results.get('combined', {})
    
    print("\nCombined Performance (Average of Temporal and Spatial):")
    for event in ['Event_1', 'Event_2']:
        avg_c = combined.get(f'{event}_avg_c_index')
        avg_ibs = combined.get(f'{event}_avg_ibs')
        
        if avg_c is not None:
            print(f"\n{event}:")
            print(f"  Average C-index: {avg_c:.4f}")
            print(f"    Temporal: {combined.get(f'{event}_temporal_c_index', 'N/A'):.4f}")
            print(f"    Spatial: {combined.get(f'{event}_spatial_c_index', 'N/A'):.4f}")
        
        if avg_ibs is not None:
            print(f"  Average IBS: {avg_ibs:.4f}")
            print(f"    Temporal: {combined.get(f'{event}_temporal_ibs', 'N/A'):.4f}")
            print(f"    Spatial: {combined.get(f'{event}_spatial_ibs', 'N/A'):.4f}")
    
    print("\n" + "="*60)