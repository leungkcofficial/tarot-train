"""
Step to evaluate different ensemble combinations.
"""

import numpy as np
import pandas as pd
import h5py
import os
from typing import Dict, Annotated
from itertools import combinations
from zenml import step
from src.evaluation_metrics import calculate_all_metrics


@step(enable_cache=False)
def evaluate_ensemble_combinations(
    temporal_cif_all: np.ndarray,
    spatial_cif_all: np.ndarray,
    y_temporal_test: pd.DataFrame,
    y_spatial_test: pd.DataFrame,
    model_info: Dict,
    max_combinations: int = 100
) -> Annotated[Dict, "evaluation_results"]:
    """
    Evaluate different ensemble combinations.
    
    Args:
        temporal_cif_all: All temporal predictions (24, 2, 5, n_samples)
        spatial_cif_all: All spatial predictions (24, 2, 5, n_samples)
        y_temporal_test: Temporal test labels
        y_spatial_test: Spatial test labels
        model_info: Model information dictionary
        max_combinations: Maximum number of combinations to evaluate (default: 100)
        
    Returns:
        Dictionary containing evaluation results
    """
    print("\nGenerating ensemble combinations...")
    
    # Time points for evaluation
    TIME_POINTS = np.array([365, 730, 1095, 1460, 1825])
    
    # Generate all possible combinations
    all_combinations = []
    combination_info = []
    
    for length in range(2, 25):  # From 2 to 24 models
        for combo in combinations(range(24), length):
            all_combinations.append(combo)
            combination_info.append({
                'combination_id': len(all_combinations) - 1,
                'n_models': length,
                'model_indices': list(combo)
            })
    
    print(f"Total combinations possible: {len(all_combinations)}")
    
    # Limit combinations for evaluation
    n_combinations = min(max_combinations, len(all_combinations))
    print(f"Evaluating {n_combinations} combinations...")
    
    # Evaluate each combination
    results = []
    
    for i in range(n_combinations):
        if i % 10 == 0:
            print(f"  Progress: {i}/{n_combinations}")
        
        combo = all_combinations[i]
        
        # Average the selected models
        temporal_ensemble = np.mean(temporal_cif_all[list(combo), :, :, :], axis=0)
        spatial_ensemble = np.mean(spatial_cif_all[list(combo), :, :, :], axis=0)
        
        # Evaluate temporal
        temporal_metrics = calculate_all_metrics(
            times=y_temporal_test['time'].values,
            events=y_temporal_test['event'].values,
            predictions=temporal_ensemble,
            time_points=TIME_POINTS
        )
        
        # Evaluate spatial
        spatial_metrics = calculate_all_metrics(
            times=y_spatial_test['time'].values,
            events=y_spatial_test['event'].values,
            predictions=spatial_ensemble,
            time_points=TIME_POINTS
        )
        
        # Store results - including event-specific IBS
        result = {
            'combination_id': i,
            'n_models': len(combo),
            'model_indices': list(combo),
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
        results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    output_dir = "results/final_deploy/ensemble_evaluation"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to H5
    output_path = os.path.join(output_dir, "ensemble_evaluation_results.h5")
    with h5py.File(output_path, 'w') as f:
        # Save results dataframe
        f.create_dataset('combination_ids', data=results_df['combination_id'].values)
        f.create_dataset('n_models', data=results_df['n_models'].values)
        f.create_dataset('temporal_ibs', data=results_df['temporal_ibs'].values)
        f.create_dataset('temporal_cidx_event1', data=results_df['temporal_cidx_event1'].values)
        f.create_dataset('temporal_cidx_event2', data=results_df['temporal_cidx_event2'].values)
        f.create_dataset('temporal_nll', data=results_df['temporal_nll'].values)
        f.create_dataset('spatial_ibs', data=results_df['spatial_ibs'].values)
        f.create_dataset('spatial_cidx_event1', data=results_df['spatial_cidx_event1'].values)
        f.create_dataset('spatial_cidx_event2', data=results_df['spatial_cidx_event2'].values)
        f.create_dataset('spatial_nll', data=results_df['spatial_nll'].values)
        
        # Save model indices for each combination
        for i in range(n_combinations):
            combo = all_combinations[i]
            f.create_dataset(f'combination_{i}_indices', data=np.array(combo))
    
    # Also save as CSV for easy viewing
    results_df.to_csv(os.path.join(output_dir, "ensemble_evaluation_results.csv"), index=False)
    
    print(f"\nResults saved to: {output_path}")
    
    # Find best combinations
    print("\nBest combinations by metric:")
    print(f"  Best temporal IBS: Combination {results_df.loc[results_df['temporal_ibs'].idxmin(), 'combination_id']} (IBS={results_df['temporal_ibs'].min():.4f})")
    print(f"  Best spatial IBS: Combination {results_df.loc[results_df['spatial_ibs'].idxmin(), 'combination_id']} (IBS={results_df['spatial_ibs'].min():.4f})")
    print(f"  Best temporal C-index (Event 1): Combination {results_df.loc[results_df['temporal_cidx_event1'].idxmax(), 'combination_id']} (C-index={results_df['temporal_cidx_event1'].max():.4f})")
    print(f"  Best spatial C-index (Event 1): Combination {results_df.loc[results_df['spatial_cidx_event1'].idxmax(), 'combination_id']} (C-index={results_df['spatial_cidx_event1'].max():.4f})")
    
    # Create evaluation results dictionary
    evaluation_results = {
        'results_df': results_df,
        'output_path': output_path,
        'n_combinations_evaluated': len(results_df),
        'best_temporal_ibs': float(results_df['temporal_ibs'].min()),
        'best_spatial_ibs': float(results_df['spatial_ibs'].min()),
        'best_temporal_cidx_event1': float(results_df['temporal_cidx_event1'].max()),
        'best_spatial_cidx_event1': float(results_df['spatial_cidx_event1'].max()),
        'all_combinations_info': combination_info[:n_combinations]
    }
    
    # Save summary
    summary = {
        'n_combinations_evaluated': evaluation_results['n_combinations_evaluated'],
        'best_temporal_ibs': evaluation_results['best_temporal_ibs'],
        'best_spatial_ibs': evaluation_results['best_spatial_ibs'],
        'best_temporal_cidx_event1': evaluation_results['best_temporal_cidx_event1'],
        'best_spatial_cidx_event1': evaluation_results['best_spatial_cidx_event1']
    }
    
    import json
    with open(os.path.join(output_dir, "evaluation_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    return evaluation_results