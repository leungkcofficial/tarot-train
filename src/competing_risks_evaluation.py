#!/usr/bin/env python3
"""
Competing risks evaluation functions for DeepHit models.
"""

import numpy as np
import pandas as pd
import torch
from typing import Tuple, Dict, Any
from pycox.models import DeepHit
from pycox.evaluation import EvalSurv


def extract_cause_specific_predictions(
    cif_predictions,
    time_grid: np.ndarray,
    cause: int = 1
) -> pd.DataFrame:
    """
    Extract cause-specific cumulative incidence predictions from DeepHit CIF output.
    
    Args:
        cif_predictions: CIF predictions array/tensor of shape (num_causes, num_time_points, num_samples)
        time_grid: Array of time points
        cause: Cause index (1-based, so cause=1 for first cause, cause=2 for second cause)
        
    Returns:
        DataFrame with time points as index and samples as columns
    """
    # Convert to 0-based indexing
    cause_idx = cause - 1
    
    # Handle both torch tensors and numpy arrays
    if hasattr(cif_predictions, 'cpu'):
        cif_numpy = cif_predictions.cpu().numpy()
    else:
        cif_numpy = cif_predictions
    
    # Extract predictions for the specific cause
    # Shape: (num_time_points, num_samples)
    cause_predictions = cif_numpy[cause_idx, :, :]
    
    # Create DataFrame
    predictions_df = pd.DataFrame(
        cause_predictions,
        index=time_grid,
        columns=range(cause_predictions.shape[1])
    )
    
    return predictions_df


def evaluate_competing_risks_model(
    model: DeepHit,
    x_data: torch.Tensor,
    durations: np.ndarray,
    events: np.ndarray,
    time_grid: np.ndarray,
    optimization_metric: str = 'cidx'
) -> Dict[str, Any]:
    """
    Evaluate a DeepHit model for competing risks.
    
    Args:
        model: Trained DeepHit model
        x_data: Input data tensor
        durations: Array of event times
        events: Array of event indicators (0=censored, 1=event1, 2=event2)
        time_grid: Array of time points for evaluation
        optimization_metric: Metric to optimize ('cidx', 'brs', 'loglik')
        
    Returns:
        Dictionary containing evaluation results for each cause
    """
    # Get CIF predictions
    with torch.no_grad():
        cif_predictions = model.predict_cif(x_data)
    
    print(f"CIF predictions shape: {cif_predictions.shape}")
    
    # Initialize results
    results = {
        'cause_1': {},
        'cause_2': {},
        'combined': {}
    }
    
    # Handle both torch tensors and numpy arrays
    if hasattr(cif_predictions, 'cpu'):
        cif_numpy = cif_predictions.cpu().numpy()
    else:
        cif_numpy = cif_predictions
    
    # Extract CIF for each cause using your suggested approach
    cif1 = pd.DataFrame(cif_numpy[0], index=time_grid)  # Cause 1 CIF
    cif2 = pd.DataFrame(cif_numpy[1], index=time_grid)  # Cause 2 CIF
    
    # Evaluate each cause separately using your method
    for cause_idx, (cause, cif_df) in enumerate([(1, cif1), (2, cif2)], 1):
        print(f"\n--- Evaluating Cause {cause} ---")
        
        # Create binary events for this cause (cause vs all others including censoring)
        binary_events = (events == cause).astype(int)
        
        print(f"Cause {cause} event rate: {binary_events.mean():.3f}")
        print(f"Cause {cause} CIF shape: {cif_df.shape}")
        
        try:
            # Use your suggested approach: EvalSurv(1-cif, durations, events == cause)
            print(f"Creating EvalSurv for Cause {cause} using 1-CIF...")
            
            # Convert CIF to survival probabilities: survival = 1 - CIF
            survival_df = 1 - cif_df
            
            # Ensure valid range [0, 1] and monotonically decreasing
            survival_df = survival_df.clip(0, 1)
            
            # Vectorized monotonic enforcement
            survival_values = survival_df.values
            survival_values = np.minimum.accumulate(survival_values, axis=0)
            survival_df = pd.DataFrame(survival_values, index=survival_df.index, columns=survival_df.columns)
            
            print(f"CIF range: [{cif_df.min().min():.3f}, {cif_df.max().max():.3f}]")
            print(f"Survival range: [{survival_df.min().min():.3f}, {survival_df.max().max():.3f}]")
            
            # Create EvalSurv object
            ev = EvalSurv(
                survival_df,
                durations,
                binary_events,
                censor_surv='km'
            )
            
            # Calculate metrics
            print(f"Calculating metrics for Cause {cause}...")
            c_index = ev.concordance_td()
            
            if optimization_metric == 'cidx':
                metric_value = c_index
                print(f"Cause {cause} C-index: {metric_value:.4f}")
            elif optimization_metric == 'brs':
                metric_value = -ev.integrated_brier_score(time_grid)
                print(f"Cause {cause} Integrated Brier Score: {-metric_value:.4f}")
            elif optimization_metric == 'loglik':
                metric_value = -ev.integrated_nbll(time_grid)
                print(f"Cause {cause} Log-likelihood: {metric_value:.4f}")
            else:
                metric_value = c_index
                print(f"Cause {cause} C-index (default): {metric_value:.4f}")
            
            results[f'cause_{cause}'] = {
                'metric_value': metric_value,
                'c_index': c_index,
                'predictions': cif_df  # Store original CIF predictions
            }
            
        except Exception as e:
            print(f"Error evaluating cause {cause}: {e}")
            results[f'cause_{cause}'] = {
                'metric_value': 0.0,
                'c_index': 0.0,
                'predictions': cause_predictions,
                'error': str(e)
            }
    
    # Calculate combined metric (average of both causes)
    cause1_metric = results['cause_1'].get('metric_value', 0.0)
    cause2_metric = results['cause_2'].get('metric_value', 0.0)
    combined_metric = (cause1_metric + cause2_metric) / 2
    
    results['combined'] = {
        'metric_value': combined_metric,
        'cause_1_metric': cause1_metric,
        'cause_2_metric': cause2_metric
    }
    
    print(f"\nCombined metric (average): {combined_metric:.4f}")
    
    return results


def save_competing_risks_predictions(
    cif_predictions,
    time_grid: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray,
    save_path: str,
    metadata_path: str
) -> None:
    """
    Save competing risks predictions in (2, 5, n_samples) format.
    
    Args:
        cif_predictions: CIF predictions array/tensor of shape (num_causes, num_time_points, num_samples)
        time_grid: Array of time points
        durations: Array of event times
        events: Array of event indicators
        save_path: Path to save predictions (HDF5 format)
        metadata_path: Path to save metadata (CSV format)
    """
    import h5py
    
    # Handle both torch tensors and numpy arrays
    if hasattr(cif_predictions, 'cpu'):
        cif_numpy = cif_predictions.cpu().numpy()
    else:
        cif_numpy = cif_predictions
    
    num_causes, num_time_points, num_samples = cif_numpy.shape
    
    # Save predictions in the preferred (2, 5, n_samples) format
    with h5py.File(save_path, 'w') as f:
        # Save the full 3D array
        f.create_dataset('predictions', data=cif_numpy)
        
        # Save cause-specific predictions for easier access
        f.create_dataset('cause_1_predictions', data=cif_numpy[0, :, :])  # Shape: (5, n_samples)
        f.create_dataset('cause_2_predictions', data=cif_numpy[1, :, :])  # Shape: (5, n_samples)
        
        # Save indices and metadata
        f.create_dataset('time_grid', data=time_grid)
        f.create_dataset('columns', data=np.arange(num_samples))
        
        # Save metadata
        metadata_group = f.create_group('metadata')
        metadata_group.create_dataset('durations', data=durations)
        metadata_group.create_dataset('events', data=events)
        metadata_group.create_dataset('time_grid', data=time_grid)
        metadata_group.create_dataset('num_causes', data=num_causes)
        metadata_group.create_dataset('num_time_points', data=num_time_points)
        metadata_group.create_dataset('num_samples', data=num_samples)
    
    # Save metadata CSV
    metadata_df = pd.DataFrame({
        'duration': durations,
        'event': events
    })
    metadata_df.to_csv(metadata_path, index=False)
    
    print(f"Saved competing risks predictions: {save_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Predictions shape: {cif_numpy.shape}")
    print(f"Format: (causes={num_causes}, time_points={num_time_points}, samples={num_samples})")
    print(f"Cause 1 shape: {cif_numpy[0, :, :].shape}")
    print(f"Cause 2 shape: {cif_numpy[1, :, :].shape}")