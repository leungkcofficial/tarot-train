"""
Evaluate the full ensemble (all 24 model groups) with comprehensive metrics.
Generates event-specific evaluations including Brier score, log-likelihood, 
calibration plots, and decision curve analysis.
"""

import numpy as np
import pandas as pd
import pickle
import h5py
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import evaluation functions
from src.metric_calculator import (
    calculate_metrics,
    plot_calibration,
    plot_competing_risks_calibration
)

# Import DCA functions
from src.dca import (
    risk_at_horizon,
    ipcw_weights,
    decision_curve,
    plot_decision_curve,
    plot_decision_curves_subplots
)


def create_full_ensemble_cif(output_dir='results/full_ensemble'):
    """
    Create CIF arrays for the full ensemble using all 24 model groups.
    
    Returns:
        temporal_ensemble_cif: (2, 5, n_samples) array for temporal test set
        spatial_ensemble_cif: (2, 5, n_samples) array for spatial test set
    """
    print("="*80)
    print("CREATING FULL ENSEMBLE CIF")
    print("="*80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load stacked predictions
    print("\nLoading stacked predictions...")
    with h5py.File('results/final_deploy/stacked_predictions/temporal_stacked_cif.h5', 'r') as f:
        temporal_stacked = f['stacked_cif'][:]
        print(f"Temporal stacked shape: {temporal_stacked.shape}")
    
    with h5py.File('results/final_deploy/stacked_predictions/spatial_stacked_cif.h5', 'r') as f:
        spatial_stacked = f['stacked_cif'][:]
        print(f"Spatial stacked shape: {spatial_stacked.shape}")
    
    # Create full ensemble by averaging all 24 models
    print("\nCreating full ensemble (averaging all 24 models)...")
    temporal_ensemble_cif = np.mean(temporal_stacked, axis=0)
    spatial_ensemble_cif = np.mean(spatial_stacked, axis=0)
    
    print(f"Temporal ensemble CIF shape: {temporal_ensemble_cif.shape}")
    print(f"Spatial ensemble CIF shape: {spatial_ensemble_cif.shape}")
    
    # Save ensemble CIFs
    print("\nSaving ensemble CIFs...")
    with h5py.File(os.path.join(output_dir, 'temporal_ensemble_cif.h5'), 'w') as f:
        f.create_dataset('ensemble_cif', data=temporal_ensemble_cif)
        f.attrs['n_models'] = 24
        f.attrs['method'] = 'simple_average'
        f.attrs['created'] = datetime.now().isoformat()
    
    with h5py.File(os.path.join(output_dir, 'spatial_ensemble_cif.h5'), 'w') as f:
        f.create_dataset('ensemble_cif', data=spatial_ensemble_cif)
        f.attrs['n_models'] = 24
        f.attrs['method'] = 'simple_average'
        f.attrs['created'] = datetime.now().isoformat()
    
    print(f"Saved ensemble CIFs to {output_dir}")
    
    return temporal_ensemble_cif, spatial_ensemble_cif


def evaluate_event_specific_metrics(cif_array, labels, dataset_name, output_dir):
    """
    Evaluate event-specific metrics for the full ensemble.
    
    Args:
        cif_array: (2, 5, n_samples) array of CIF predictions
        labels: DataFrame with 'duration' and 'event' columns
        dataset_name: 'temporal' or 'spatial'
        output_dir: Directory to save results
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING {dataset_name.upper()} DATASET")
    print(f"{'='*80}")
    
    # Extract labels
    event_times = labels['event_times']
    event_indicators = labels['event_indicators']
    
    # Store original continuous durations for C-index and DCA calculation
    continuous_durations = event_times.copy()
    
    # Discretize durations for DeepHit calibration evaluation
    # Time grid: [365, 730, 1095, 1460, 1825]
    time_grid = np.array([365, 730, 1095, 1460, 1825])
    time_points = time_grid  # For consistency
    
    # Create discrete durations (0-4 corresponding to the 5 time points)
    discrete_durations = np.zeros_like(event_times)
    for i, duration in enumerate(event_times):
        if duration <= time_grid[0]:
            discrete_durations[i] = 0
        elif duration <= time_grid[1]:
            discrete_durations[i] = 1
        elif duration <= time_grid[2]:
            discrete_durations[i] = 2
        elif duration <= time_grid[3]:
            discrete_durations[i] = 3
        else:
            discrete_durations[i] = 4
    
    print(f"Discretized durations to 5 bins: {np.unique(discrete_durations)}")
    print(f"Original duration range: [{continuous_durations.min():.1f}, {continuous_durations.max():.1f}]")
    
    print(f"Number of samples: {len(event_times)}")
    print(f"Event 1 count: {np.sum(event_indicators == 1)} ({np.mean(event_indicators == 1)*100:.1f}%)")
    print(f"Event 2 count: {np.sum(event_indicators == 2)} ({np.mean(event_indicators == 2)*100:.1f}%)")
    print(f"Censored count: {np.sum(event_indicators == 0)} ({np.mean(event_indicators == 0)*100:.1f}%)")
    
    # Create dataset-specific output directory
    dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Initialize results dictionary
    results = {
        'dataset': dataset_name,
        'n_samples': len(event_times),
        'event_rates': {
            'event_1': float(np.mean(event_indicators == 1)),
            'event_2': float(np.mean(event_indicators == 2)),
            'censored': float(np.mean(event_indicators == 0))
        },
        'metrics': {}
    }
    
    # 1. Calculate Brier Scores
    print("\n1. Calculating Brier Scores...")
    brier_scores = {'event_1': {}, 'event_2': {}}
    
    for event_idx, event_name in enumerate(['event_1', 'event_2']):
        event_num = event_idx + 1
        print(f"\n   Event {event_num}:")
        
        for t_idx, t in enumerate(time_points):
            # Get predictions at this time point
            predictions = cif_array[event_idx, t_idx, :]
            
            # Calculate Brier score using continuous durations
            at_risk = continuous_durations >= t
            observed = ((continuous_durations <= t) & (event_indicators == event_num)).astype(float)
            
            if np.sum(at_risk) > 0:
                brier = np.mean((predictions[at_risk] - observed[at_risk])**2)
                brier_scores[event_name][f't_{int(t)}'] = float(brier)
                print(f"      t={t}: {brier:.4f}")
            else:
                brier_scores[event_name][f't_{int(t)}'] = np.nan
                print(f"      t={t}: No samples at risk")
    
    results['metrics']['brier_scores'] = brier_scores
    
    # 2. Calculate Integrated Brier Score (IBS)
    print("\n2. Calculating Integrated Brier Scores...")
    ibs_scores = {}
    
    for event_idx, event_name in enumerate(['event_1', 'event_2']):
        event_num = event_idx + 1
        
        # Calculate IBS by averaging Brier scores
        bs_values = [v for k, v in brier_scores[event_name].items() if not np.isnan(v)]
        if bs_values:
            ibs = np.mean(bs_values)
            ibs_scores[event_name] = float(ibs)
            print(f"   Event {event_num} IBS: {ibs:.4f}")
        else:
            ibs_scores[event_name] = np.nan
    
    results['metrics']['ibs'] = ibs_scores
    
    # 3. Calculate Concordance Index
    print("\n3. Calculating Concordance Index...")
    from sksurv.metrics import concordance_index_censored
    
    c_indices = {}
    for event_idx, event_name in enumerate(['event_1', 'event_2']):
        event_num = event_idx + 1
        
        # Get predictions at the last time point (5-year horizon)
        predictions_5y = cif_array[event_idx, -1, :]
        
        # Create binary event indicator for this specific event
        binary_events = (event_indicators == event_num).astype(int)
        
        # Calculate C-index using continuous durations (not discretized)
        # This avoids ties and gives more accurate discrimination
        c_index = concordance_index_censored(
            binary_events.astype(bool),
            continuous_durations,  # Use original continuous durations
            predictions_5y
        )[0]
        
        c_indices[event_name] = float(c_index)
        print(f"   Event {event_num} C-index: {c_index:.4f}")
    
    results['metrics']['c_index'] = c_indices
    
    # 4. Calculate Log-Likelihood at each time point
    print("\n4. Calculating Log-Likelihood at each time point...")
    log_likelihoods = {'event_1': {}, 'event_2': {}}
    
    for event_idx, event_name in enumerate(['event_1', 'event_2']):
        event_num = event_idx + 1
        print(f"\n   Event {event_num}:")
        
        for t_idx, t in enumerate(time_points):
            # Get predictions at this time point
            predictions = cif_array[event_idx, t_idx, :]
            
            # Get events that occurred before or at this time
            event_mask = (event_indicators == event_num) & (continuous_durations <= t)
            
            if np.sum(event_mask) > 0:
                # Get predictions for these events
                event_probs = predictions[event_mask]
                
                # Calculate log-likelihood (avoiding log(0))
                event_probs = np.clip(event_probs, 1e-10, 1-1e-10)
                ll = np.mean(np.log(event_probs))
                log_likelihoods[event_name][f't_{int(t)}'] = float(ll)
                print(f"      t={t}: {ll:.4f}")
            else:
                log_likelihoods[event_name][f't_{int(t)}'] = np.nan
                print(f"      t={t}: No events")
    
    results['metrics']['log_likelihood'] = log_likelihoods
    
    # 5. Generate Calibration Plots
    print("\n5. Generating Calibration Plots...")
    
    # Use the standard calibration plot function for each event
    time_horizons = [365, 730, 1095, 1460, 1825]
    
    for event_idx, event_name in enumerate(['event_1', 'event_2']):
        event_num = event_idx + 1
        
        # Create DataFrame with predictions for this event at standard time horizons
        pred_df = pd.DataFrame()
        
        for horizon in time_horizons:
            # Find closest time index
            t_idx = np.argmin(np.abs(time_points - horizon))
            pred_df[horizon] = cif_array[event_idx, t_idx, :]
        
        # Create calibration plot using standard function with discretized durations
        calibration_plot_path = os.path.join(dataset_dir, f'{event_name}_calibration.png')
        plot_calibration(
            pred_df,
            discrete_durations,  # Use discretized durations for calibration
            event_indicators,
            time_horizons,
            calibration_plot_path,
            model_type='deephit',
            event_of_interest=event_num
        )
        print(f"   Saved {event_name} calibration plot to {calibration_plot_path}")
    
    # 6. Decision Curve Analysis
    print("\n6. Performing Decision Curve Analysis...")
    
    # DCA for each event at all time points
    dca_time_points = [365, 730, 1095, 1460, 1825]  # All 5 time horizons
    
    for event_idx, event_name in enumerate(['event_1', 'event_2']):
        event_num = event_idx + 1
        
        # Prepare DCA data for all time horizons
        dca_data_list = []
        
        for horizon in dca_time_points:
            # Find closest time index
            t_idx = np.argmin(np.abs(time_points - horizon))
            
            # Get predictions for this event and time
            predictions = cif_array[event_idx, t_idx, :]
            
            # For competing risks, we need to create binary event indicators
            # Event of interest = 1, everything else (other event or censored) = 0
            binary_events = (event_indicators == event_num).astype(int)
            
            # Calculate decision curve using continuous durations
            # DCA uses continuous time for proper censoring weights
            dca_result = decision_curve(
                predictions,
                continuous_durations,  # Use original continuous durations
                binary_events,
                horizon
            )
            
            # Add to list with proper formatting
            dca_data_list.append({
                'thresholds': dca_result['thresholds'],
                'net_benefit': dca_result['nb_model'],
                'net_benefit_all': dca_result['nb_treat_all'],
                'net_benefit_none': dca_result['nb_treat_none'],
                'label': f'Full Ensemble',
                'horizon': f'{horizon}'
            })
        
        # Create DCA plot using standard function
        dca_plot_path = os.path.join(dataset_dir, f'{event_name}_decision_curves.png')
        plot_decision_curves_subplots(dca_data_list, dca_plot_path)
        print(f"   Saved {event_name} decision curves to {dca_plot_path}")
    
    # 7. Create comprehensive metrics plots over time
    print("\n7. Creating comprehensive metrics plots over time...")
    
    for event_idx, event_name in enumerate(['event_1', 'event_2']):
        # Extract values for each metric
        brier_values = []
        ll_values = []
        for t in time_points:
            brier_values.append(brier_scores[event_name][f't_{int(t)}'])
            ll_values.append(log_likelihoods[event_name][f't_{int(t)}'])
        
        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        # 1. Plot Brier scores
        ax1.plot(time_points, brier_values, 'o-', color='blue', linewidth=2, markersize=8, label='Brier Score')
        # Add confidence intervals (using bootstrap approximation)
        n_events = np.sum(event_indicators == (event_idx + 1))
        if n_events > 0:
            # Approximate CI for Brier score
            se_brier = np.sqrt(np.array(brier_values) * (1 - np.array(brier_values)) / n_events)
            brier_lower = np.array(brier_values) - 1.96 * se_brier
            brier_upper = np.array(brier_values) + 1.96 * se_brier
            ax1.fill_between(time_points, brier_lower, brier_upper, color='blue', alpha=0.2)
        
        ax1.set_ylabel('Brier Score')
        ax1.set_title(f'Brier Score by Time Horizon - {event_name.replace("_", " ").title()}')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(loc='upper right')
        
        # 2. Plot C-index (time-specific)
        # For competing risks, we calculate C-index at each time point
        c_index_values = []
        for t_idx, t in enumerate(time_points):
            predictions = cif_array[event_idx, t_idx, :]
            binary_events = (event_indicators == (event_idx + 1)).astype(int)
            try:
                from sksurv.metrics import concordance_index_ipcw
                # Create structured array for scikit-survival
                y_train = np.array([(binary_events[i], continuous_durations[i])
                                   for i in range(len(binary_events))],
                                  dtype=[('event', bool), ('time', float)])
                c_idx = concordance_index_ipcw(y_train, y_train, predictions, tau=t)[0]
                c_index_values.append(c_idx)
            except:
                # Fallback to regular C-index
                c_idx = concordance_index_censored(
                    binary_events.astype(bool),
                    continuous_durations,
                    predictions
                )[0]
                c_index_values.append(c_idx)
        
        ax2.plot(time_points, c_index_values, 'o-', color='green', linewidth=2, markersize=8, label='C-index (IPCW)')
        # Add confidence intervals
        if n_events > 0:
            # Approximate CI for C-index
            se_c = 0.5 / np.sqrt(n_events)  # Rough approximation
            c_lower = np.array(c_index_values) - 1.96 * se_c
            c_upper = np.array(c_index_values) + 1.96 * se_c
            ax2.fill_between(time_points, c_lower, c_upper, color='green', alpha=0.2)
        
        ax2.set_ylabel('C-index (IPCW)')
        ax2.set_title(f'Time-specific Concordance Index by Time Horizon - {event_name.replace("_", " ").title()}')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(loc='lower right')
        ax2.set_ylim([0.5, 1.0])
        
        # 3. Plot Log-likelihood
        ax3.plot(time_points, ll_values, 'o-', color='red', linewidth=2, markersize=8, label='Log-Likelihood')
        # Add confidence intervals
        if n_events > 0:
            se_ll = 1.0 / np.sqrt(n_events)
            ll_lower = [ll - 1.96 * se_ll if not np.isnan(ll) else np.nan for ll in ll_values]
            ll_upper = [ll + 1.96 * se_ll if not np.isnan(ll) else np.nan for ll in ll_values]
            ax3.fill_between(time_points, ll_lower, ll_upper, color='red', alpha=0.2)
        
        ax3.set_xlabel('Time Horizon (days)')
        ax3.set_ylabel('Log-Likelihood')
        ax3.set_title(f'Log-Likelihood by Time Horizon - {event_name.replace("_", " ").title()}')
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.legend(loc='lower right')
        
        # Add integrated metrics as text box
        textstr = f'Integrated Metrics:\n'
        textstr += f'C-index: {c_indices[event_name]:.4f} (95% CI: {c_indices[event_name]-1.96*0.5/np.sqrt(n_events):.4f}-{c_indices[event_name]+1.96*0.5/np.sqrt(n_events):.4f})\n'
        textstr += f'IBS: {ibs_scores[event_name]:.4f} (95% CI: {ibs_scores[event_name]-1.96*np.sqrt(ibs_scores[event_name]*(1-ibs_scores[event_name])/n_events):.4f}-{ibs_scores[event_name]+1.96*np.sqrt(ibs_scores[event_name]*(1-ibs_scores[event_name])/n_events):.4f})\n'
        
        # Calculate NBLL (Negative Binomial Log-Likelihood) - approximation
        nbll = -np.nanmean([ll for ll in ll_values if not np.isnan(ll)])
        textstr += f'NBLL: {nbll:.4f} (95% CI: {nbll-1.96/np.sqrt(n_events):.4f}-{nbll+1.96/np.sqrt(n_events):.4f})'
        
        # Add text box to first subplot
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        # Save plot
        metrics_plot_path = os.path.join(dataset_dir, f'{event_name}_metrics_by_time.png')
        plt.savefig(metrics_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   Saved {event_name} comprehensive metrics plot to {metrics_plot_path}")
    
    # 8. Save results
    import json
    with open(os.path.join(dataset_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved evaluation results to {dataset_dir}/evaluation_results.json")
    
    return results


def main():
    """Main function to create full ensemble and evaluate."""
    
    output_dir = 'results/full_ensemble'
    
    # Create full ensemble CIFs
    temporal_ensemble_cif, spatial_ensemble_cif = create_full_ensemble_cif(output_dir)
    
    # Load labels
    print("\nLoading ground truth labels...")
    with open('results/final_deploy/temporal_labels.pkl', 'rb') as f:
        temporal_labels = pickle.load(f)
    
    with open('results/final_deploy/spatial_labels.pkl', 'rb') as f:
        spatial_labels = pickle.load(f)
    
    # Evaluate both datasets
    temporal_results = evaluate_event_specific_metrics(
        temporal_ensemble_cif,
        temporal_labels,
        'temporal',
        output_dir
    )
    
    spatial_results = evaluate_event_specific_metrics(
        spatial_ensemble_cif,
        spatial_labels,
        'spatial',
        output_dir
    )
    
    # Create summary report
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    print("\nTemporal Test Set:")
    print(f"  Event 1 - C-index: {temporal_results['metrics']['c_index']['event_1']:.4f}")
    print(f"  Event 2 - C-index: {temporal_results['metrics']['c_index']['event_2']:.4f}")
    print(f"  Event 1 - IBS: {temporal_results['metrics']['ibs']['event_1']:.4f}")
    print(f"  Event 2 - IBS: {temporal_results['metrics']['ibs']['event_2']:.4f}")
    
    print("\nSpatial Test Set:")
    print(f"  Event 1 - C-index: {spatial_results['metrics']['c_index']['event_1']:.4f}")
    print(f"  Event 2 - C-index: {spatial_results['metrics']['c_index']['event_2']:.4f}")
    print(f"  Event 1 - IBS: {spatial_results['metrics']['ibs']['event_1']:.4f}")
    print(f"  Event 2 - IBS: {spatial_results['metrics']['ibs']['event_2']:.4f}")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()