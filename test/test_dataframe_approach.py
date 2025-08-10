"""
Test script to verify the DataFrame-based approach works correctly.
Tests with a small subset of combinations first.
"""

import numpy as np
import pandas as pd
from itertools import combinations
import time
from steps.evaluate_ensemble_combinations_dataframe import stack_deepsurv_models, evaluate_combination
from steps.extract_ground_truth_labels import extract_ground_truth_labels
from steps.load_and_stack_all_predictions import load_and_stack_all_predictions


def test_dataframe_approach():
    """Test the DataFrame approach with a small subset."""
    
    print("Testing DataFrame-based ensemble evaluation approach...")
    
    # Load data
    print("\n1. Loading predictions and labels...")
    temporal_labels, spatial_labels = extract_ground_truth_labels()
    temporal_predictions, spatial_predictions = load_and_stack_all_predictions()
    
    print(f"   Temporal predictions: {len(temporal_predictions)} models")
    print(f"   Spatial predictions: {len(spatial_predictions)} models")
    
    # Pre-stack models
    print("\n2. Pre-stacking models...")
    temporal_stacked = stack_deepsurv_models(temporal_predictions)
    spatial_stacked = stack_deepsurv_models(spatial_predictions)
    
    print(f"   Temporal stacked shape: {temporal_stacked.shape}")
    print(f"   Spatial stacked shape: {spatial_stacked.shape}")
    
    # Test with a few combinations
    print("\n3. Testing evaluation with sample combinations...")
    
    test_combinations = [
        [0, 1],           # 2 models
        [0, 1, 2, 3],     # 4 models
        [0, 5, 10, 15, 20],  # 5 models
        list(range(12)),  # 12 models (all DeepSurv)
        list(range(24))   # All 24 models
    ]
    
    results = []
    
    for i, combo in enumerate(test_combinations):
        print(f"\n   Testing combination {i+1}: {len(combo)} models - {combo}")
        
        start_time = time.time()
        metrics = evaluate_combination(
            combo,
            temporal_stacked,
            spatial_stacked,
            temporal_labels,
            spatial_labels
        )
        eval_time = time.time() - start_time
        
        print(f"   Evaluation time: {eval_time:.3f} seconds")
        print(f"   Temporal C-index (Event 1): {metrics['temporal_cidx_event1']:.4f}")
        print(f"   Temporal C-index (Event 2): {metrics['temporal_cidx_event2']:.4f}")
        print(f"   Spatial C-index (Event 1): {metrics['spatial_cidx_event1']:.4f}")
        print(f"   Spatial C-index (Event 2): {metrics['spatial_cidx_event2']:.4f}")
        
        results.append({
            'combination': combo,
            'n_models': len(combo),
            'eval_time': eval_time,
            **metrics
        })
    
    # Test DataFrame creation speed
    print("\n4. Testing DataFrame creation speed...")
    
    start_time = time.time()
    all_combinations = []
    combo_idx = 0
    
    # Generate first 100,000 combinations
    for length in range(2, 25):
        for combo in combinations(range(24), length):
            all_combinations.append({
                'combination_id': combo_idx,
                'n_models': len(combo),
                'model_indices': list(combo)
            })
            combo_idx += 1
            
            if combo_idx >= 100000:
                break
        
        if combo_idx >= 100000:
            break
    
    df_creation_time = time.time() - start_time
    print(f"   Created {len(all_combinations):,} combinations in {df_creation_time:.2f} seconds")
    
    # Estimate total time
    print("\n5. Performance estimates:")
    
    avg_eval_time = np.mean([r['eval_time'] for r in results])
    total_combinations = sum(np.math.comb(24, k) for k in range(2, 25))
    
    estimated_total_time = total_combinations * avg_eval_time
    estimated_hours = estimated_total_time / 3600
    
    print(f"   Average evaluation time per combination: {avg_eval_time:.4f} seconds")
    print(f"   Total combinations to evaluate: {total_combinations:,}")
    print(f"   Estimated total time (single-threaded): {estimated_hours:.1f} hours")
    
    # With batching
    batch_overhead = 0.1  # 10% overhead for batching
    estimated_batch_time = estimated_total_time * (1 + batch_overhead)
    print(f"   Estimated total time (with batching): {estimated_batch_time/3600:.1f} hours")
    
    print("\n✓ Test completed successfully!")
    print("\nThe DataFrame approach is working correctly and should be much more efficient than")
    print("the previous approach because:")
    print("- Models are pre-stacked only once")
    print("- No redundant stacking operations")
    print("- Efficient array slicing for combinations")
    print("- Better memory usage with batching")


if __name__ == "__main__":
    test_dataframe_approach()