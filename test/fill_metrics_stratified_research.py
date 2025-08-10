"""
Stratified sampling for ensemble evaluation with research-focused questions.
Ensures adequate representation for specific comparisons.
"""

import numpy as np
import pandas as pd
import pickle
import h5py
import os
import time
from datetime import datetime, timedelta
from itertools import combinations
from collections import defaultdict
import json


def load_model_metadata():
    """Load model metadata to understand model characteristics."""
    # Read model config
    model_config = pd.read_csv('model_config.csv')
    
    # Create model info dictionary
    model_info = {}
    for idx, row in model_config.iterrows():
        model_id = idx  # 0-based index
        model_info[model_id] = {
            'algorithm': row['Algorithm'],
            'structure': row['Structure'],
            'balancing': row['Balancing_Method'],
            'optimization': row['Optimization_Target'],
            'event': row['Event']
        }
    
    # Map stacked indices to original model indices
    # DeepSurv: 0-11 (stacked groups)
    # DeepHit: 12-23 (individual models)
    stacked_to_original = {}
    
    # DeepSurv groups (from model_grouping_summary.md)
    deepsurv_groups = [
        [0, 2],    # Group 1: ANN + None + Concordance
        [1, 3],    # Group 2: ANN + None + Log-likelihood
        [4, 6],    # Group 3: ANN + NearMiss v1 + Concordance
        [5, 7],    # Group 4: ANN + NearMiss v1 + Log-likelihood
        [8, 10],   # Group 5: ANN + KNN + Concordance
        [9, 11],   # Group 6: ANN + KNN + Log-likelihood
        [12, 14],  # Group 7: LSTM + None + Concordance
        [13, 15],  # Group 8: LSTM + None + Log-likelihood
        [16, 18],  # Group 9: LSTM + NearMiss v3 + Concordance
        [17, 19],  # Group 10: LSTM + NearMiss v3 + Log-likelihood
        [20, 22],  # Group 11: LSTM + KNN + Concordance
        [21, 23],  # Group 12: LSTM + KNN + Log-likelihood
    ]
    
    # Map stacked indices
    for stacked_idx, original_indices in enumerate(deepsurv_groups):
        stacked_to_original[stacked_idx] = {
            'original_indices': original_indices,
            'algorithm': 'DeepSurv',
            'structure': model_info[original_indices[0]]['structure'],
            'balancing': model_info[original_indices[0]]['balancing'],
            'optimization': model_info[original_indices[0]]['optimization']
        }
    
    # DeepHit models (12-23 in stacked, 24-35 in original)
    for i in range(12):
        stacked_idx = 12 + i
        original_idx = 24 + i
        stacked_to_original[stacked_idx] = {
            'original_indices': [original_idx],
            'algorithm': 'DeepHit',
            'structure': model_info[original_idx]['structure'],
            'balancing': model_info[original_idx]['balancing'],
            'optimization': model_info[original_idx]['optimization']
        }
    
    return model_info, stacked_to_original


def categorize_combination(model_indices, stacked_to_original):
    """Categorize a combination based on model characteristics."""
    categories = {
        'n_models': len(model_indices),
        'algorithms': set(),
        'structures': set(),
        'balancing_methods': set(),
        'optimization_targets': set(),
        'has_deepsurv': False,
        'has_deephit': False,
        'has_ann': False,
        'has_lstm': False,
        'has_balanced': False,
        'has_unbalanced': False,
        'is_mixed_algorithm': False,
        'is_mixed_structure': False,
        'is_mixed_balancing': False
    }
    
    for idx in model_indices:
        info = stacked_to_original[idx]
        categories['algorithms'].add(info['algorithm'])
        categories['structures'].add(info['structure'])
        categories['balancing_methods'].add(info['balancing'])
        categories['optimization_targets'].add(info['optimization'])
        
        if info['algorithm'] == 'DeepSurv':
            categories['has_deepsurv'] = True
        else:
            categories['has_deephit'] = True
        
        if info['structure'] == 'ANN':
            categories['has_ann'] = True
        else:
            categories['has_lstm'] = True
        
        if info['balancing'] == 'None':
            categories['has_unbalanced'] = True
        else:
            categories['has_balanced'] = True
    
    # Mixed categories
    categories['is_mixed_algorithm'] = len(categories['algorithms']) > 1
    categories['is_mixed_structure'] = len(categories['structures']) > 1
    categories['is_mixed_balancing'] = len(categories['balancing_methods']) > 1
    
    # Convert sets to counts for easier analysis
    categories['n_algorithms'] = len(categories['algorithms'])
    categories['n_structures'] = len(categories['structures'])
    categories['n_balancing_methods'] = len(categories['balancing_methods'])
    
    return categories


def generate_research_focused_samples(total_samples=60000):
    """Generate samples that ensure adequate representation for research questions."""
    
    print("Loading model metadata...")
    model_info, stacked_to_original = load_model_metadata()
    
    # Define research questions and required samples
    research_categories = {
        # 1. Ensemble size analysis
        'by_size': defaultdict(list),
        
        # 2. Balanced vs Unbalanced
        'balanced_only': [],
        'unbalanced_only': [],
        'mixed_balancing': [],
        
        # 3. Algorithm comparison (DeepSurv vs DeepHit)
        'deepsurv_only': [],
        'deephit_only': [],
        'mixed_algorithm': [],
        
        # 4. Structure comparison (ANN vs LSTM)
        'ann_only': [],
        'lstm_only': [],
        'mixed_structure': [],
        
        # 5. Specific interesting combinations
        'all_balanced_mixed': [],  # Balanced models with mixed algorithms/structures
        'all_unbalanced_mixed': [],  # Unbalanced models with mixed algorithms/structures
    }
    
    # Minimum samples per category to ensure statistical significance
    min_samples_per_category = {
        'by_size': 500,  # Per size
        'balanced_only': 2000,
        'unbalanced_only': 2000,
        'mixed_balancing': 3000,
        'deepsurv_only': 2000,
        'deephit_only': 2000,
        'mixed_algorithm': 3000,
        'ann_only': 2000,
        'lstm_only': 2000,
        'mixed_structure': 3000,
        'all_balanced_mixed': 1000,
        'all_unbalanced_mixed': 1000
    }
    
    print("\nGenerating stratified samples for research questions...")
    
    # Track all sampled combinations to avoid duplicates
    all_samples = set()
    samples_by_category = defaultdict(list)
    
    # 1. First, ensure coverage for each ensemble size
    print("\n1. Sampling by ensemble size...")
    for n_models in range(1, 25):  # 1 to 24 models
        print(f"   Size {n_models}: ", end='')
        
        # Calculate how many possible combinations exist for this size
        total_possible = len(list(combinations(range(24), n_models)))
        
        # Sample proportionally but ensure minimum
        n_samples = max(min_samples_per_category['by_size'], 
                       int(total_samples * 0.3 * total_possible / 16777215))
        n_samples = min(n_samples, total_possible)  # Don't exceed possible
        
        # Generate random combinations for this size
        sampled = 0
        attempts = 0
        while sampled < n_samples and attempts < n_samples * 10:
            combo = tuple(sorted(np.random.choice(24, n_models, replace=False)))
            if combo not in all_samples:
                all_samples.add(combo)
                samples_by_category['by_size'][n_models].append(combo)
                
                # Also categorize for other research questions
                cats = categorize_combination(combo, stacked_to_original)
                
                # Balanced vs Unbalanced
                if cats['has_balanced'] and not cats['has_unbalanced']:
                    samples_by_category['balanced_only'].append(combo)
                elif cats['has_unbalanced'] and not cats['has_balanced']:
                    samples_by_category['unbalanced_only'].append(combo)
                elif cats['has_balanced'] and cats['has_unbalanced']:
                    samples_by_category['mixed_balancing'].append(combo)
                
                # Algorithm comparison
                if cats['has_deepsurv'] and not cats['has_deephit']:
                    samples_by_category['deepsurv_only'].append(combo)
                elif cats['has_deephit'] and not cats['has_deepsurv']:
                    samples_by_category['deephit_only'].append(combo)
                elif cats['has_deepsurv'] and cats['has_deephit']:
                    samples_by_category['mixed_algorithm'].append(combo)
                
                # Structure comparison
                if cats['has_ann'] and not cats['has_lstm']:
                    samples_by_category['ann_only'].append(combo)
                elif cats['has_lstm'] and not cats['has_ann']:
                    samples_by_category['lstm_only'].append(combo)
                elif cats['has_ann'] and cats['has_lstm']:
                    samples_by_category['mixed_structure'].append(combo)
                
                # Special combinations
                if cats['has_balanced'] and not cats['has_unbalanced'] and \
                   (cats['is_mixed_algorithm'] or cats['is_mixed_structure']):
                    samples_by_category['all_balanced_mixed'].append(combo)
                
                if cats['has_unbalanced'] and not cats['has_balanced'] and \
                   (cats['is_mixed_algorithm'] or cats['is_mixed_structure']):
                    samples_by_category['all_unbalanced_mixed'].append(combo)
                
                sampled += 1
            attempts += 1
        
        print(f"{sampled} samples")
    
    # 2. Ensure minimum samples for each research category
    print("\n2. Ensuring minimum samples per research category...")
    
    for category, min_required in min_samples_per_category.items():
        if category == 'by_size':
            continue  # Already handled
        
        current_count = len(samples_by_category[category])
        needed = min_required - current_count
        
        if needed > 0:
            print(f"   {category}: have {current_count}, need {needed} more")
            
            # Generate targeted samples for this category
            attempts = 0
            added = 0
            
            while added < needed and attempts < needed * 100:
                n_models = np.random.randint(2, 20)  # Focus on reasonable ensemble sizes
                combo = tuple(sorted(np.random.choice(24, n_models, replace=False)))
                
                if combo not in all_samples:
                    cats = categorize_combination(combo, stacked_to_original)
                    
                    # Check if this combination fits the category
                    fits_category = False
                    
                    if category == 'balanced_only' and cats['has_balanced'] and not cats['has_unbalanced']:
                        fits_category = True
                    elif category == 'unbalanced_only' and cats['has_unbalanced'] and not cats['has_balanced']:
                        fits_category = True
                    elif category == 'mixed_balancing' and cats['has_balanced'] and cats['has_unbalanced']:
                        fits_category = True
                    elif category == 'deepsurv_only' and cats['has_deepsurv'] and not cats['has_deephit']:
                        fits_category = True
                    elif category == 'deephit_only' and cats['has_deephit'] and not cats['has_deepsurv']:
                        fits_category = True
                    elif category == 'mixed_algorithm' and cats['has_deepsurv'] and cats['has_deephit']:
                        fits_category = True
                    elif category == 'ann_only' and cats['has_ann'] and not cats['has_lstm']:
                        fits_category = True
                    elif category == 'lstm_only' and cats['has_lstm'] and not cats['has_ann']:
                        fits_category = True
                    elif category == 'mixed_structure' and cats['has_ann'] and cats['has_lstm']:
                        fits_category = True
                    
                    if fits_category:
                        all_samples.add(combo)
                        samples_by_category[category].append(combo)
                        added += 1
                
                attempts += 1
        else:
            print(f"   {category}: {current_count} samples (sufficient)")
    
    # 3. Fill remaining quota with diverse samples
    remaining = total_samples - len(all_samples)
    if remaining > 0:
        print(f"\n3. Adding {remaining} diverse samples to reach {total_samples} total...")
        
        added = 0
        while added < remaining:
            n_models = np.random.choice(range(2, 20), p=np.exp(-np.arange(18)/5)/np.sum(np.exp(-np.arange(18)/5)))
            combo = tuple(sorted(np.random.choice(24, n_models, replace=False)))
            
            if combo not in all_samples:
                all_samples.add(combo)
                added += 1
    
    # Convert to list and create DataFrame
    print(f"\nTotal unique samples: {len(all_samples)}")
    
    # Create combinations DataFrame
    combinations_list = []
    for i, combo in enumerate(all_samples):
        combinations_list.append({
            'combination_id': i,
            'n_models': len(combo),
            'model_indices': list(combo)
        })
    
    combinations_df = pd.DataFrame(combinations_list)
    
    # Save sampling statistics
    sampling_stats = {
        'total_samples': len(all_samples),
        'by_ensemble_size': {size: len(combos) for size, combos in samples_by_category['by_size'].items()},
        'by_category': {cat: len(combos) for cat, combos in samples_by_category.items() if cat != 'by_size'},
        'research_questions': {
            'balanced_vs_unbalanced': {
                'balanced_only': len(samples_by_category['balanced_only']),
                'unbalanced_only': len(samples_by_category['unbalanced_only']),
                'mixed': len(samples_by_category['mixed_balancing'])
            },
            'algorithm_comparison': {
                'deepsurv_only': len(samples_by_category['deepsurv_only']),
                'deephit_only': len(samples_by_category['deephit_only']),
                'mixed': len(samples_by_category['mixed_algorithm'])
            },
            'structure_comparison': {
                'ann_only': len(samples_by_category['ann_only']),
                'lstm_only': len(samples_by_category['lstm_only']),
                'mixed': len(samples_by_category['mixed_structure'])
            }
        }
    }
    
    with open('results/ensemble_checkpoints/sampling_statistics.json', 'w') as f:
        json.dump(sampling_stats, f, indent=2)
    
    print("\nSampling statistics saved to: results/ensemble_checkpoints/sampling_statistics.json")
    
    return combinations_df, samples_by_category


def main():
    """Main evaluation function with research-focused sampling."""
    
    # Generate research-focused samples
    print("="*80)
    print("RESEARCH-FOCUSED ENSEMBLE EVALUATION")
    print("="*80)
    
    combinations_df, samples_by_category = generate_research_focused_samples(60000)
    
    # Save combinations for reference
    combinations_file = 'results/ensemble_checkpoints/research_combinations.csv'
    combinations_df.to_csv(combinations_file, index=False)
    print(f"\nSaved {len(combinations_df)} combinations to: {combinations_file}")
    
    # Now run evaluation (similar to sample_based but with our custom combinations)
    output_file = 'results/ensemble_checkpoints/evaluation_results_research.csv'
    
    print("\n" + "="*80)
    print("STARTING EVALUATION")
    print("="*80)
    print(f"Output file: {output_file}")
    print(f"Total combinations: {len(combinations_df):,}")
    
    # Load data
    print("\nLoading stacked CIF predictions...")
    with h5py.File('results/final_deploy/stacked_predictions/temporal_stacked_cif.h5', 'r') as f:
        temporal_stacked = f['stacked_cif'][:]
    
    with h5py.File('results/final_deploy/stacked_predictions/spatial_stacked_cif.h5', 'r') as f:
        spatial_stacked = f['stacked_cif'][:]
    
    print("Loading ground truth labels...")
    with open('results/final_deploy/temporal_labels.pkl', 'rb') as f:
        temporal_labels = pickle.load(f)
    
    with open('results/final_deploy/spatial_labels.pkl', 'rb') as f:
        spatial_labels = pickle.load(f)
    
    # Import evaluation functions
    from src.evaluation_metrics import calculate_all_metrics
    
    # Time points for evaluation
    time_points = np.array([365, 730, 1095, 1460, 1825])
    
    # Initialize results
    results = []
    
    print("\nEvaluating combinations...")
    start_time = time.time()
    
    for idx, row in combinations_df.iterrows():
        if idx % 100 == 0:
            elapsed = time.time() - start_time
            rate = idx / elapsed if elapsed > 0 and idx > 0 else 0
            remaining = len(combinations_df) - idx
            eta_seconds = remaining / rate if rate > 0 else 0
            eta_str = str(timedelta(seconds=int(eta_seconds)))
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                  f"Progress: {idx}/{len(combinations_df)} ({idx/len(combinations_df)*100:.1f}%) | "
                  f"Rate: {rate:.1f} combos/sec | ETA: {eta_str}")
        
        # Get model indices
        model_indices = row['model_indices']
        
        # Create ensemble
        temporal_ensemble = np.mean(temporal_stacked[model_indices], axis=0)
        spatial_ensemble = np.mean(spatial_stacked[model_indices], axis=0)
        
        # Calculate metrics
        temporal_metrics = calculate_all_metrics(
            temporal_labels['event_times'],
            temporal_labels['event_indicators'],
            temporal_ensemble,
            time_points
        )
        
        spatial_metrics = calculate_all_metrics(
            spatial_labels['event_times'],
            spatial_labels['event_indicators'],
            spatial_ensemble,
            time_points
        )
        
        # Store results
        result = {
            'combination_id': row['combination_id'],
            'n_models': row['n_models'],
            'model_indices': str(model_indices),
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
            'spatial_nll': spatial_metrics['nll'],
            'temporal_cidx_avg': (temporal_metrics['cidx_event1'] + temporal_metrics['cidx_event2']) / 2,
            'spatial_cidx_avg': (spatial_metrics['cidx_event1'] + spatial_metrics['cidx_event2']) / 2,
            'overall_cidx_avg': ((temporal_metrics['cidx_event1'] + temporal_metrics['cidx_event2'] + 
                                 spatial_metrics['cidx_event1'] + spatial_metrics['cidx_event2']) / 4)
        }
        results.append(result)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    
    total_time = time.time() - start_time
    print(f"\nEvaluation completed in {str(timedelta(seconds=int(total_time)))}")
    print(f"Results saved to: {output_file}")
    
    # Analyze results by research questions
    print("\n" + "="*80)
    print("RESEARCH QUESTION ANALYSIS")
    print("="*80)
    
    analyze_research_questions(results_df, combinations_df, samples_by_category)


def analyze_research_questions(results_df, combinations_df, samples_by_category):
    """Analyze results for specific research questions."""
    
    # Load model metadata
    _, stacked_to_original = load_model_metadata()
    
    # Add categories to results
    for idx, row in results_df.iterrows():
        model_indices = eval(row['model_indices'])
        cats = categorize_combination(model_indices, stacked_to_original)
        
        for key, value in cats.items():
            if isinstance(value, set):
                results_df.loc[idx, key] = len(value)
            else:
                results_df.loc[idx, key] = value
    
    print("\n1. ENSEMBLE SIZE ANALYSIS")
    print("-" * 50)
    size_analysis = results_df.groupby('n_models')['overall_cidx_avg'].agg(['mean', 'std', 'max', 'count'])
    print(size_analysis.sort_values('mean', ascending=False).head(10))
    
    print("\n2. BALANCED vs UNBALANCED MODELS")
    print("-" * 50)
    balanced_only = results_df[results_df['has_balanced'] & ~results_df['has_unbalanced']]
    unbalanced_only = results_df[~results_df['has_balanced'] & results_df['has_unbalanced']]
    mixed_balance = results_df[results_df['has_balanced'] & results_df['has_unbalanced']]
    
    print(f"Balanced only - Mean C-index: {balanced_only['overall_cidx_avg'].mean():.4f} (n={len(balanced_only)})")
    print(f"Unbalanced only - Mean C-index: {unbalanced_only['overall_cidx_avg'].mean():.4f} (n={len(unbalanced_only)})")
    print(f"Mixed balancing - Mean C-index: {mixed_balance['overall_cidx_avg'].mean():.4f} (n={len(mixed_balance)})")
    
    print("\n3. ALGORITHM COMPARISON (DeepSurv vs DeepHit)")
    print("-" * 50)
    deepsurv_only = results_df[results_df['has_deepsurv'] & ~results_df['has_deephit']]
    deephit_only = results_df[~results_df['has_deepsurv'] & results_df['has_deephit']]
    mixed_algo = results_df[results_df['has_deepsurv'] & results_df['has_deephit']]
    
    print(f"DeepSurv only - Mean C-index: {deepsurv_only['overall_cidx_avg'].mean():.4f} (n={len(deepsurv_only)})")
    print(f"DeepHit only - Mean C-index: {deephit_only['overall_cidx_avg'].mean():.4f} (n={len(deephit_only)})")
    print(f"Mixed algorithms - Mean C-index: {mixed_algo['overall_cidx_avg'].mean():.4f} (n={len(mixed_algo)})")
    
    print("\n4. STRUCTURE COMPARISON (ANN vs LSTM)")
    print("-" * 50)
    ann_only = results_df[results_df['has_ann'] & ~results_df['has_lstm']]
    lstm_only = results_df[~results_df['has_ann'] & results_df['has_lstm']]
    mixed_struct = results_df[results_df['has_ann'] & results_df['has_lstm']]
    
    print(f"ANN only - Mean C-index: {ann_only['overall_cidx_avg'].mean():.4f} (n={len(ann_only)})")
    print(f"LSTM only - Mean C-index: {lstm_only['overall_cidx_avg'].mean():.4f} (n={len(lstm_only)})")
    print(f"Mixed structures - Mean C-index: {mixed_struct['overall_cidx_avg'].mean():.4f} (n={len(mixed_struct)})")
    
    print("\n5. TOP 10 BEST COMBINATIONS")
    print("-" * 50)
    top_10 = results_df.nlargest(10, 'overall_cidx_avg')
    
    for idx, row in top_10.iterrows():
        model_indices = eval(row['model_indices'])
        cats = categorize_combination(model_indices, stacked_to_original)
        
        print(f"\nRank {idx+1}: C-index = {row['overall_cidx_avg']:.4f}")
        print(f"  Models: {row['n_models']} | Indices: {row['model_indices']}")
        print(f"  Algorithms: {cats['algorithms']}")
        print(f"  Structures: {cats['structures']}")
        print(f"  Balancing: {cats['balancing_methods']}")
    
    # Save detailed analysis
    analysis_file = 'results/ensemble_checkpoints/research_analysis.txt'
    with open(analysis_file, 'w') as f:
        f.write("DETAILED RESEARCH ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        # More detailed statistical tests could be added here
        # For example: t-tests between groups, effect sizes, etc.
    
    print(f"\nDetailed analysis saved to: {analysis_file}")


if __name__ == "__main__":
    main()