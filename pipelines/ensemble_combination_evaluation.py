"""
Ensemble Combination Evaluation Pipeline

This script evaluates all possible combinations of 24 CIF arrays from:
- 12 DeepSurv models for Event 1
- 12 DeepSurv models for Event 2  
- 12 DeepHit models (already in CIF format)

For each combination, it performs bootstrap evaluation with:
- Integrated Brier Score (IBS)
- Concordance Index (C-index)
- Negative Log-Likelihood (NLL)

Results include mean and 95% CI for each metric across bootstraps.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import combinations
from typing import Dict, List, Tuple, Any
import warnings
from tqdm import tqdm
import pickle
from scipy import stats

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation_metrics import (
    concordance_index_censored,
    integrated_brier_score,
    negative_log_likelihood
)

class EnsembleCombinationEvaluator:
    """Evaluates all possible combinations of model predictions"""
    
    def __init__(self, 
                 predictions_dir: str,
                 output_dir: str,
                 n_bootstraps: int = 10,
                 random_seed: int = 42):
        """
        Initialize evaluator
        
        Args:
            predictions_dir: Directory containing individual model predictions
            output_dir: Directory to save evaluation results
            n_bootstraps: Number of bootstrap iterations
            random_seed: Random seed for reproducibility
        """
        self.predictions_dir = predictions_dir
        self.output_dir = output_dir
        self.n_bootstraps = n_bootstraps
        self.random_seed = random_seed
        
        # Time points for evaluation
        self.time_points = np.array([365, 730, 1095, 1460, 1825])
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set random seed
        np.random.seed(random_seed)
        
    def load_all_predictions(self) -> Dict[str, Dict[str, Any]]:
        """Load all 24 CIF arrays and their metadata"""
        print("Loading all model predictions...")
        
        predictions = {}
        
        # Load DeepSurv Event 1 predictions (models 1-12)
        for i in range(1, 13):
            pred_file = os.path.join(self.predictions_dir, f'model{i}_predictions.npz')
            if os.path.exists(pred_file):
                data = np.load(pred_file)
                predictions[f'deepsurv_event1_{i}'] = {
                    'cif': data['cif_predictions'],  # Shape: (5, n_samples)
                    'model_id': i,
                    'model_type': 'deepsurv',
                    'event': 1,
                    'metadata': json.loads(str(data['metadata']))
                }
                print(f"  Loaded DeepSurv Event 1 Model {i}: shape {data['cif_predictions'].shape}")
        
        # Load DeepSurv Event 2 predictions (models 13-24)
        for i in range(13, 25):
            pred_file = os.path.join(self.predictions_dir, f'model{i}_predictions.npz')
            if os.path.exists(pred_file):
                data = np.load(pred_file)
                predictions[f'deepsurv_event2_{i}'] = {
                    'cif': data['cif_predictions'],  # Shape: (5, n_samples)
                    'model_id': i,
                    'model_type': 'deepsurv',
                    'event': 2,
                    'metadata': json.loads(str(data['metadata']))
                }
                print(f"  Loaded DeepSurv Event 2 Model {i}: shape {data['cif_predictions'].shape}")
        
        # Load DeepHit predictions (models 25-36)
        for i in range(25, 37):
            pred_file = os.path.join(self.predictions_dir, f'model{i}_predictions.npz')
            if os.path.exists(pred_file):
                data = np.load(pred_file)
                # DeepHit predictions are already in CIF format: (2, 5, n_samples)
                # We'll use both events for evaluation
                predictions[f'deephit_{i}'] = {
                    'cif': data['predictions'],  # Shape: (2, 5, n_samples)
                    'model_id': i,
                    'model_type': 'deephit',
                    'event': 'both',
                    'metadata': json.loads(str(data['metadata']))
                }
                print(f"  Loaded DeepHit Model {i}: shape {data['predictions'].shape}")
        
        print(f"\nTotal models loaded: {len(predictions)}")
        return predictions
    
    def load_test_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load test data for evaluation"""
        # Load from the test data saved during prediction
        test_file = os.path.join(self.predictions_dir, 'test_data.npz')
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Test data not found at {test_file}")
        
        data = np.load(test_file)
        X_test = data['X_test']
        y_test = data['y_test']  # Structured array with 'time' and 'event'
        
        # Extract time and event arrays
        times = y_test['time']
        events = y_test['event']
        
        return X_test, times, events
    
    def create_ensemble_prediction(self, 
                                 predictions: Dict[str, Dict[str, Any]], 
                                 model_keys: List[str]) -> np.ndarray:
        """
        Create ensemble prediction from selected models
        
        Args:
            predictions: All model predictions
            model_keys: Keys of models to include in ensemble
            
        Returns:
            Ensemble CIF array of shape (2, 5, n_samples)
        """
        # Separate by event type
        event1_preds = []
        event2_preds = []
        
        for key in model_keys:
            pred = predictions[key]
            
            if pred['model_type'] == 'deepsurv':
                if pred['event'] == 1:
                    event1_preds.append(pred['cif'])  # (5, n_samples)
                else:  # event == 2
                    event2_preds.append(pred['cif'])  # (5, n_samples)
            else:  # deephit
                # DeepHit predictions are (2, 5, n_samples)
                event1_preds.append(pred['cif'][0])  # Event 1: (5, n_samples)
                event2_preds.append(pred['cif'][1])  # Event 2: (5, n_samples)
        
        # Average predictions for each event
        if event1_preds:
            event1_ensemble = np.mean(event1_preds, axis=0)  # (5, n_samples)
        else:
            # If no event 1 predictions, use zeros
            event1_ensemble = np.zeros_like(event2_preds[0])
            
        if event2_preds:
            event2_ensemble = np.mean(event2_preds, axis=0)  # (5, n_samples)
        else:
            # If no event 2 predictions, use zeros
            event2_ensemble = np.zeros_like(event1_preds[0])
        
        # Stack to create (2, 5, n_samples)
        ensemble = np.stack([event1_ensemble, event2_ensemble], axis=0)
        
        return ensemble
    
    def bootstrap_evaluate(self,
                         cif_predictions: np.ndarray,
                         times: np.ndarray,
                         events: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Perform bootstrap evaluation of predictions
        
        Args:
            cif_predictions: CIF array of shape (2, 5, n_samples)
            times: Observed times
            events: Observed events (0, 1, or 2)
            
        Returns:
            Dictionary with bootstrap results for each metric
        """
        n_samples = len(times)
        
        # Storage for bootstrap results
        ibs_scores = []
        cidx_event1_scores = []
        cidx_event2_scores = []
        nll_scores = []
        
        for b in range(self.n_bootstraps):
            # Bootstrap sample
            idx = np.random.choice(n_samples, size=n_samples, replace=True)
            
            boot_times = times[idx]
            boot_events = events[idx]
            boot_cif = cif_predictions[:, :, idx]
            
            # Integrated Brier Score
            try:
                ibs = integrated_brier_score(
                    boot_times, boot_events, boot_cif, self.time_points
                )
                ibs_scores.append(ibs)
            except Exception as e:
                print(f"IBS calculation failed in bootstrap {b}: {e}")
                ibs_scores.append(np.nan)
            
            # Concordance Index for Event 1
            event1_mask = boot_events != 2  # Not event 2
            if np.sum(event1_mask) > 10:  # Need enough samples
                try:
                    cidx1 = concordance_index_censored(
                        boot_events[event1_mask] == 1,
                        boot_times[event1_mask],
                        -boot_cif[0, -1, idx][event1_mask]  # Use last time point, negative for risk
                    )[0]
                    cidx_event1_scores.append(cidx1)
                except Exception as e:
                    print(f"C-index Event 1 calculation failed in bootstrap {b}: {e}")
                    cidx_event1_scores.append(np.nan)
            else:
                cidx_event1_scores.append(np.nan)
            
            # Concordance Index for Event 2
            event2_mask = boot_events != 1  # Not event 1
            if np.sum(event2_mask) > 10:  # Need enough samples
                try:
                    cidx2 = concordance_index_censored(
                        boot_events[event2_mask] == 2,
                        boot_times[event2_mask],
                        -boot_cif[1, -1, idx][event2_mask]  # Use last time point, negative for risk
                    )[0]
                    cidx_event2_scores.append(cidx2)
                except Exception as e:
                    print(f"C-index Event 2 calculation failed in bootstrap {b}: {e}")
                    cidx_event2_scores.append(np.nan)
            else:
                cidx_event2_scores.append(np.nan)
            
            # Negative Log-Likelihood
            try:
                nll = negative_log_likelihood(
                    boot_times, boot_events, boot_cif, self.time_points
                )
                nll_scores.append(nll)
            except Exception as e:
                print(f"NLL calculation failed in bootstrap {b}: {e}")
                nll_scores.append(np.nan)
        
        # Calculate statistics
        results = {}
        
        # Helper function to calculate mean and CI
        def calc_stats(scores):
            scores = [s for s in scores if not np.isnan(s)]
            if not scores:
                return {'mean': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan, 'std': np.nan}
            
            mean = np.mean(scores)
            std = np.std(scores)
            ci_lower, ci_upper = np.percentile(scores, [2.5, 97.5])
            
            return {
                'mean': mean,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'std': std,
                'scores': scores
            }
        
        results['ibs'] = calc_stats(ibs_scores)
        results['cidx_event1'] = calc_stats(cidx_event1_scores)
        results['cidx_event2'] = calc_stats(cidx_event2_scores)
        results['nll'] = calc_stats(nll_scores)
        
        return results
    
    def evaluate_all_combinations(self, max_combinations: int = None):
        """
        Evaluate all possible combinations of models
        
        Args:
            max_combinations: Maximum number of combinations to evaluate (for testing)
        """
        # Load predictions and test data
        predictions = self.load_all_predictions()
        X_test, times, events = self.load_test_data()
        
        # Get all model keys
        all_keys = list(predictions.keys())
        n_models = len(all_keys)
        
        print(f"\nEvaluating combinations of {n_models} models...")
        print(f"Total possible combinations: {2**n_models - 1}")
        
        # Results storage
        all_results = []
        
        # Generate and evaluate combinations
        total_combinations = 0
        
        # Iterate through all possible combination sizes
        for r in range(1, n_models + 1):
            print(f"\nEvaluating combinations of size {r}...")
            
            combo_count = 0
            for combo in tqdm(combinations(all_keys, r), 
                            desc=f"Size {r}",
                            total=min(len(list(combinations(range(n_models), r))), 
                                    max_combinations or float('inf'))):
                
                # Create ensemble prediction
                ensemble_pred = self.create_ensemble_prediction(predictions, list(combo))
                
                # Bootstrap evaluation
                eval_results = self.bootstrap_evaluate(ensemble_pred, times, events)
                
                # Store results
                result = {
                    'combination_id': total_combinations,
                    'n_models': len(combo),
                    'model_keys': list(combo),
                    'model_ids': [predictions[k]['model_id'] for k in combo],
                    'model_types': [predictions[k]['model_type'] for k in combo],
                    'timestamp': datetime.now().isoformat(),
                    'metrics': eval_results
                }
                
                all_results.append(result)
                
                total_combinations += 1
                combo_count += 1
                
                # Save intermediate results every 1000 combinations
                if total_combinations % 1000 == 0:
                    self.save_intermediate_results(all_results)
                
                # Check if we've reached max combinations
                if max_combinations and total_combinations >= max_combinations:
                    print(f"\nReached maximum combinations limit: {max_combinations}")
                    break
            
            if max_combinations and total_combinations >= max_combinations:
                break
            
            print(f"  Evaluated {combo_count} combinations of size {r}")
        
        print(f"\nTotal combinations evaluated: {total_combinations}")
        
        # Save final results
        self.save_final_results(all_results)
        
        # Find best combinations
        self.analyze_best_combinations(all_results)
        
        return all_results
    
    def save_intermediate_results(self, results: List[Dict]):
        """Save intermediate results during evaluation"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f'intermediate_results_{timestamp}.pkl')
        
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
    
    def save_final_results(self, results: List[Dict]):
        """Save final evaluation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as pickle for full data
        pkl_file = os.path.join(self.output_dir, f'ensemble_evaluation_results_{timestamp}.pkl')
        with open(pkl_file, 'wb') as f:
            pickle.dump(results, f)
        
        # Save summary as CSV
        summary_data = []
        for r in results:
            summary = {
                'combination_id': r['combination_id'],
                'n_models': r['n_models'],
                'model_ids': ','.join(map(str, r['model_ids'])),
                'ibs_mean': r['metrics']['ibs']['mean'],
                'ibs_ci_lower': r['metrics']['ibs']['ci_lower'],
                'ibs_ci_upper': r['metrics']['ibs']['ci_upper'],
                'cidx_event1_mean': r['metrics']['cidx_event1']['mean'],
                'cidx_event1_ci_lower': r['metrics']['cidx_event1']['ci_lower'],
                'cidx_event1_ci_upper': r['metrics']['cidx_event1']['ci_upper'],
                'cidx_event2_mean': r['metrics']['cidx_event2']['mean'],
                'cidx_event2_ci_lower': r['metrics']['cidx_event2']['ci_lower'],
                'cidx_event2_ci_upper': r['metrics']['cidx_event2']['ci_upper'],
                'nll_mean': r['metrics']['nll']['mean'],
                'nll_ci_lower': r['metrics']['nll']['ci_lower'],
                'nll_ci_upper': r['metrics']['nll']['ci_upper']
            }
            summary_data.append(summary)
        
        summary_df = pd.DataFrame(summary_data)
        csv_file = os.path.join(self.output_dir, f'ensemble_evaluation_summary_{timestamp}.csv')
        summary_df.to_csv(csv_file, index=False)
        
        print(f"\nResults saved to:")
        print(f"  Full data: {pkl_file}")
        print(f"  Summary: {csv_file}")
    
    def analyze_best_combinations(self, results: List[Dict], top_k: int = 10):
        """Analyze and report best performing combinations"""
        print(f"\n{'='*80}")
        print("BEST PERFORMING ENSEMBLE COMBINATIONS")
        print(f"{'='*80}")
        
        # Sort by different metrics
        metrics_to_analyze = [
            ('ibs', 'ascending'),  # Lower is better
            ('cidx_event1', 'descending'),  # Higher is better
            ('cidx_event2', 'descending'),  # Higher is better
            ('nll', 'ascending')  # Lower is better
        ]
        
        for metric, order in metrics_to_analyze:
            print(f"\nTop {top_k} by {metric.upper()} (best {order}):")
            print("-" * 60)
            
            # Sort results
            if order == 'ascending':
                sorted_results = sorted(results, 
                                      key=lambda x: x['metrics'][metric]['mean'])
            else:
                sorted_results = sorted(results, 
                                      key=lambda x: x['metrics'][metric]['mean'], 
                                      reverse=True)
            
            # Display top results
            for i, r in enumerate(sorted_results[:top_k]):
                mean = r['metrics'][metric]['mean']
                ci_lower = r['metrics'][metric]['ci_lower']
                ci_upper = r['metrics'][metric]['ci_upper']
                
                print(f"{i+1}. Combination {r['combination_id']} ({r['n_models']} models)")
                print(f"   {metric.upper()}: {mean:.4f} (95% CI: {ci_lower:.4f} - {ci_upper:.4f})")
                print(f"   Models: {r['model_ids']}")
                
                # Show model composition
                model_types = {}
                for mt in r['model_types']:
                    model_types[mt] = model_types.get(mt, 0) + 1
                print(f"   Composition: {model_types}")
                print()
        
        # Find overall best (balanced across metrics)
        print(f"\n{'='*80}")
        print("OVERALL BEST COMBINATIONS (balanced across all metrics)")
        print(f"{'='*80}")
        
        # Normalize metrics for fair comparison
        for r in results:
            # Calculate normalized scores (0-1, where 1 is best)
            r['normalized_scores'] = {}
            
            # IBS and NLL: lower is better, so invert
            ibs_values = [x['metrics']['ibs']['mean'] for x in results if not np.isnan(x['metrics']['ibs']['mean'])]
            if ibs_values:
                ibs_min, ibs_max = min(ibs_values), max(ibs_values)
                r['normalized_scores']['ibs'] = 1 - (r['metrics']['ibs']['mean'] - ibs_min) / (ibs_max - ibs_min) if ibs_max > ibs_min else 1
            
            nll_values = [x['metrics']['nll']['mean'] for x in results if not np.isnan(x['metrics']['nll']['mean'])]
            if nll_values:
                nll_min, nll_max = min(nll_values), max(nll_values)
                r['normalized_scores']['nll'] = 1 - (r['metrics']['nll']['mean'] - nll_min) / (nll_max - nll_min) if nll_max > nll_min else 1
            
            # C-index: higher is better
            cidx1_values = [x['metrics']['cidx_event1']['mean'] for x in results if not np.isnan(x['metrics']['cidx_event1']['mean'])]
            if cidx1_values:
                cidx1_min, cidx1_max = min(cidx1_values), max(cidx1_values)
                r['normalized_scores']['cidx_event1'] = (r['metrics']['cidx_event1']['mean'] - cidx1_min) / (cidx1_max - cidx1_min) if cidx1_max > cidx1_min else 1
            
            cidx2_values = [x['metrics']['cidx_event2']['mean'] for x in results if not np.isnan(x['metrics']['cidx_event2']['mean'])]
            if cidx2_values:
                cidx2_min, cidx2_max = min(cidx2_values), max(cidx2_values)
                r['normalized_scores']['cidx_event2'] = (r['metrics']['cidx_event2']['mean'] - cidx2_min) / (cidx2_max - cidx2_min) if cidx2_max > cidx2_min else 1
            
            # Calculate overall score (average of normalized scores)
            valid_scores = [v for v in r['normalized_scores'].values() if not np.isnan(v)]
            r['overall_score'] = np.mean(valid_scores) if valid_scores else 0
        
        # Sort by overall score
        sorted_by_overall = sorted(results, key=lambda x: x['overall_score'], reverse=True)
        
        for i, r in enumerate(sorted_by_overall[:top_k]):
            print(f"\n{i+1}. Combination {r['combination_id']} (Overall Score: {r['overall_score']:.4f})")
            print(f"   Models ({r['n_models']}): {r['model_ids']}")
            print(f"   IBS: {r['metrics']['ibs']['mean']:.4f} (normalized: {r['normalized_scores'].get('ibs', np.nan):.4f})")
            print(f"   C-index Event 1: {r['metrics']['cidx_event1']['mean']:.4f} (normalized: {r['normalized_scores'].get('cidx_event1', np.nan):.4f})")
            print(f"   C-index Event 2: {r['metrics']['cidx_event2']['mean']:.4f} (normalized: {r['normalized_scores'].get('cidx_event2', np.nan):.4f})")
            print(f"   NLL: {r['metrics']['nll']['mean']:.4f} (normalized: {r['normalized_scores'].get('nll', np.nan):.4f})")


def main():
    """Main execution function"""
    # Configuration
    predictions_dir = "results/final_deploy/predictions"
    output_dir = "results/final_deploy/ensemble_evaluation"
    
    # For testing, you might want to limit combinations
    # Set to None to evaluate ALL combinations (warning: 2^24 - 1 = 16,777,215 combinations!)
    max_combinations = 1000  # Start with first 1000 for testing
    
    # Create evaluator
    evaluator = EnsembleCombinationEvaluator(
        predictions_dir=predictions_dir,
        output_dir=output_dir,
        n_bootstraps=10,
        random_seed=42
    )
    
    # Run evaluation
    print("Starting ensemble combination evaluation...")
    print(f"Max combinations to evaluate: {max_combinations if max_combinations else 'ALL'}")
    
    results = evaluator.evaluate_all_combinations(max_combinations=max_combinations)
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()