"""
Optimized step to evaluate ALL ensemble combinations using DataFrame-based approach.
Pre-stacks models and uses efficient slicing for evaluation.
"""

import numpy as np
import pandas as pd
import h5py
import os
import pickle
from typing import Dict, Annotated, Tuple, List
from itertools import combinations
from zenml import step
from src.evaluation_metrics import calculate_all_metrics
import time
from datetime import datetime
import gc


def stack_deepsurv_models(all_predictions: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Stack DeepSurv models according to model_grouping_summary.md.
    Returns array of shape (24, 2, 5, n_samples).
    """
    # Group mapping based on model_grouping_summary.md
    group_mapping = {
        1: [1, 2],    # Group 1: Models 1 (Event 1) + 2 (Event 2)
        2: [3, 4],    # Group 2: Models 3 (Event 1) + 4 (Event 2)
        3: [5, 6],    # Group 3: Models 5 (Event 1) + 6 (Event 2)
        4: [7, 8],    # Group 4: Models 7 (Event 1) + 8 (Event 2)
        5: [9, 10],   # Group 5: Models 9 (Event 1) + 10 (Event 2)
        6: [11, 12],  # Group 6: Models 11 (Event 1) + 12 (Event 2)
        7: [13, 14],  # Group 7: Models 13 (Event 1) + 14 (Event 2)
        8: [15, 16],  # Group 8: Models 15 (Event 1) + 16 (Event 2)
        9: [17, 18],  # Group 9: Models 17 (Event 1) + 18 (Event 2)
        10: [19, 20], # Group 10: Models 19 (Event 1) + 20 (Event 2)
        11: [21, 22], # Group 11: Models 21 (Event 1) + 22 (Event 2)
        12: [23, 24], # Group 12: Models 23 (Event 1) + 24 (Event 2)
    }
    
    # Time points to extract (in days): 365, 730, 1095, 1460, 1825
    time_indices = [364, 729, 1094, 1459, 1824]  # 0-indexed
    
    # Get number of samples from first model
    first_model_key = next(iter(all_predictions.keys()))
    if len(all_predictions[first_model_key].shape) == 2:
        # DeepSurv model
        n_samples = all_predictions[first_model_key].shape[1]
    else:
        # DeepHit model
        n_samples = all_predictions[first_model_key].shape[2]
    
    # Initialize stacked array: (24, 2, 5, n_samples)
    # 24 = 12 DeepSurv groups + 12 DeepHit models
    stacked_predictions = np.zeros((24, 2, 5, n_samples))
    
    # Stack DeepSurv models (groups 0-11)
    for group_idx, (group_num, model_ids) in enumerate(group_mapping.items()):
        event1_model_id = model_ids[0]
        event2_model_id = model_ids[1]
        
        # Get predictions for both events
        event1_pred = all_predictions[f"model_{event1_model_id}"]  # Shape: (1825, n_samples)
        event2_pred = all_predictions[f"model_{event2_model_id}"]  # Shape: (1825, n_samples)
        
        # Extract predictions at specific time points
        for t_idx, time_idx in enumerate(time_indices):
            stacked_predictions[group_idx, 0, t_idx, :] = event1_pred[time_idx, :]  # Event 1
            stacked_predictions[group_idx, 1, t_idx, :] = event2_pred[time_idx, :]  # Event 2
    
    # Add DeepHit models (groups 12-23)
    for i in range(12):
        model_id = 25 + i  # DeepHit models are 25-36
        deephit_pred = all_predictions[f"model_{model_id}"]  # Shape: (2, 5, n_samples)
        stacked_predictions[12 + i, :, :, :] = deephit_pred
    
    return stacked_predictions


def evaluate_combination(
    combo_indices: List[int],
    temporal_stacked: np.ndarray,
    spatial_stacked: np.ndarray,
    temporal_labels: Dict[str, np.ndarray],
    spatial_labels: Dict[str, np.ndarray]
) -> Dict[str, float]:
    """
    Evaluate a single combination by slicing and averaging predictions.
    
    Args:
        combo_indices: List of model indices to include in ensemble
        temporal_stacked: Pre-stacked temporal predictions (24, 2, 5, n_samples)
        spatial_stacked: Pre-stacked spatial predictions (24, 2, 5, n_samples)
        temporal_labels: Ground truth for temporal test set
        spatial_labels: Ground truth for spatial test set
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Slice predictions for this combination
    temporal_combo = temporal_stacked[combo_indices, :, :, :]  # (n_models, 2, 5, n_samples)
    spatial_combo = spatial_stacked[combo_indices, :, :, :]    # (n_models, 2, 5, n_samples)
    
    # Average across models
    temporal_ensemble = np.mean(temporal_combo, axis=0)  # (2, 5, n_samples)
    spatial_ensemble = np.mean(spatial_combo, axis=0)    # (2, 5, n_samples)
    
    # Time points for evaluation (in days)
    time_points = np.array([365, 730, 1095, 1460, 1825])
    
    # Calculate metrics
    temporal_metrics = calculate_all_metrics(
        temporal_labels['event_times'],      # times
        temporal_labels['event_indicators'],  # events
        temporal_ensemble,                    # predictions
        time_points                          # time_points
    )
    
    spatial_metrics = calculate_all_metrics(
        spatial_labels['event_times'],       # times
        spatial_labels['event_indicators'],   # events
        spatial_ensemble,                     # predictions
        time_points                          # time_points
    )
    
    return {
        'temporal_ibs': temporal_metrics['ibs'],
        'temporal_ibs_event1': temporal_metrics['ibs_event1'],
        'temporal_ibs_event2': temporal_metrics['ibs_event2'],
        'temporal_cidx_event1': temporal_metrics['cidx_event1'],
        'temporal_cidx_event2': temporal_metrics['cidx_event2'],
        'temporal_nll': temporal_metrics['nll'],
        'spatial_ibs': spatial_metrics['ibs'],
        'spatial_ibs_event1': spatial_metrics['ibs_event1'],
        'spatial_ibs_event2': spatial_metrics['ibs_event2'],
        'spatial_cidx_event1': spatial_metrics['cidx_event1'],
        'spatial_cidx_event2': spatial_metrics['cidx_event2'],
        'spatial_nll': spatial_metrics['nll']
    }


@step
def evaluate_ensemble_combinations_dataframe(
    temporal_predictions: Annotated[Dict[str, np.ndarray], "temporal_predictions"],
    spatial_predictions: Annotated[Dict[str, np.ndarray], "spatial_predictions"],
    temporal_labels: Annotated[Dict[str, np.ndarray], "temporal_labels"],
    spatial_labels: Annotated[Dict[str, np.ndarray], "spatial_labels"],
    checkpoint_dir: str = "results/ensemble_checkpoints",
    batch_size: int = 100000
) -> Annotated[pd.DataFrame, "ensemble_results"]:
    """
    Evaluate all possible ensemble combinations using DataFrame-based approach.
    
    This implementation:
    1. Pre-stacks all models into (24, 2, 5, n_samples) arrays
    2. Creates DataFrame with all combinations
    3. Iterates through rows to calculate metrics
    4. Supports checkpointing for resumability
    """
    print(f"Starting ensemble evaluation with DataFrame approach at {datetime.now()}")
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_dir, "evaluation_checkpoint.pkl")
    results_file = os.path.join(checkpoint_dir, "evaluation_results.csv")
    
    # Check if we're resuming from a checkpoint
    start_row = 0
    if os.path.exists(checkpoint_file):
        print(f"Found checkpoint file, attempting to resume...")
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
                start_row = checkpoint_data['last_completed_row'] + 1
                print(f"Resuming from row {start_row}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}, starting from beginning")
            start_row = 0
    
    # Pre-stack all models
    print("Pre-stacking all models...")
    temporal_stacked = stack_deepsurv_models(temporal_predictions)
    spatial_stacked = stack_deepsurv_models(spatial_predictions)
    print(f"Stacked shapes - Temporal: {temporal_stacked.shape}, Spatial: {spatial_stacked.shape}")
    
    # Generate all combinations and create DataFrame
    if start_row == 0:
        print("Generating all combinations...")
        all_combinations = []
        combo_idx = 0
        
        # Generate combinations of all possible lengths (2 to 24)
        for length in range(2, 25):
            for combo in combinations(range(24), length):
                all_combinations.append({
                    'combination_id': combo_idx,
                    'n_models': len(combo),
                    'model_indices': list(combo),
                    'temporal_ibs': np.nan,
                    'temporal_ibs_event1': np.nan,
                    'temporal_ibs_event2': np.nan,
                    'temporal_cidx_event1': np.nan,
                    'temporal_cidx_event2': np.nan,
                    'temporal_nll': np.nan,
                    'spatial_ibs': np.nan,
                    'spatial_ibs_event1': np.nan,
                    'spatial_ibs_event2': np.nan,
                    'spatial_cidx_event1': np.nan,
                    'spatial_cidx_event2': np.nan,
                    'spatial_nll': np.nan
                })
                combo_idx += 1
                
                # Print progress every 100k combinations
                if combo_idx % 100000 == 0:
                    print(f"Generated {combo_idx:,} combinations...")
        
        print(f"Total combinations generated: {len(all_combinations):,}")
        
        # Create DataFrame
        df = pd.DataFrame(all_combinations)
        
        # Save initial DataFrame
        df.to_csv(results_file, index=False)
        print(f"Saved initial DataFrame to {results_file}")
    else:
        # Load existing DataFrame
        print(f"Loading existing DataFrame from {results_file}")
        df = pd.read_csv(results_file)
        # Convert model_indices from string to list
        df['model_indices'] = df['model_indices'].apply(eval)
    
    total_combinations = len(df)
    print(f"Total combinations to evaluate: {total_combinations:,}")
    
    # Process in batches
    start_time = time.time()
    
    for batch_start in range(start_row, total_combinations, batch_size):
        batch_end = min(batch_start + batch_size, total_combinations)
        batch_time_start = time.time()
        
        print(f"\nProcessing batch {batch_start:,} to {batch_end:,}")
        
        # Evaluate combinations in this batch
        for idx in range(batch_start, batch_end):
            if idx % 10000 == 0 and idx > batch_start:
                elapsed = time.time() - start_time
                rate = (idx - start_row) / elapsed
                eta_seconds = (total_combinations - idx) / rate if rate > 0 else 0
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                
                print(f"Progress: {idx:,}/{total_combinations:,} ({idx/total_combinations*100:.2f}%) "
                      f"Rate: {rate:.1f} combos/sec, ETA: {eta_str}")
            
            # Get model indices for this combination
            model_indices = df.loc[idx, 'model_indices']
            
            # Evaluate this combination
            metrics = evaluate_combination(
                model_indices,
                temporal_stacked,
                spatial_stacked,
                temporal_labels,
                spatial_labels
            )
            
            # Update DataFrame
            for key, value in metrics.items():
                df.loc[idx, key] = value
        
        # Save batch results
        df.to_csv(results_file, index=False)
        
        # Update checkpoint
        with open(checkpoint_file, 'wb') as f:
            pickle.dump({
                'last_completed_row': batch_end - 1,
                'timestamp': datetime.now()
            }, f)
        
        batch_time = time.time() - batch_time_start
        print(f"Batch completed in {batch_time:.1f} seconds")
        
        # Force garbage collection
        gc.collect()
    
    # Final save
    df.to_csv(results_file, index=False)
    
    # Calculate final statistics
    total_time = time.time() - start_time
    print(f"\nEvaluation completed!")
    print(f"Total time: {str(datetime.timedelta(seconds=int(total_time)))}")
    print(f"Total combinations evaluated: {total_combinations:,}")
    print(f"Average rate: {total_combinations/total_time:.1f} combinations/second")
    
    # Find best combinations
    print("\nFinding best combinations...")
    
    # Best by temporal C-index (average of both events)
    df['temporal_cidx_avg'] = (df['temporal_cidx_event1'] + df['temporal_cidx_event2']) / 2
    df['spatial_cidx_avg'] = (df['spatial_cidx_event1'] + df['spatial_cidx_event2']) / 2
    df['overall_cidx_avg'] = (df['temporal_cidx_avg'] + df['spatial_cidx_avg']) / 2
    
    # Sort by overall C-index
    df_sorted = df.sort_values('overall_cidx_avg', ascending=False)
    
    print("\nTop 10 combinations by overall C-index:")
    print(df_sorted[['combination_id', 'n_models', 'overall_cidx_avg', 
                     'temporal_cidx_avg', 'spatial_cidx_avg']].head(10))
    
    # Clean up checkpoint file
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"\nRemoved checkpoint file: {checkpoint_file}")
    
    return df