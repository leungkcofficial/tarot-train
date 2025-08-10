"""
Script to compare ensemble model performance with KFRE and null models.
This script:
1. Loads ensemble CIF predictions
2. Loads ground truth labels
3. Calculates KFRE predictions (or loads if already calculated)
4. Fits Aalen-Johansen null model
5. Performs quantile-based analysis
6. Generates comparison plots
7. Saves results to JSON
"""

import pandas as pd
import numpy as np
import h5py
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime
from lifelines import AalenJohansenFitter
from sklearn.metrics import brier_score_loss
from lifelines.utils import concordance_index
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import DCA functions
from src.dca import decision_curve, risk_at_horizon

def load_ensemble_cif(dataset_type='temporal'):
    """
    Load ensemble CIF arrays from HDF5 files.
    
    Args:
        dataset_type: 'temporal' or 'spatial'
    
    Returns:
        numpy array of shape (2, 5, n_samples)
    """
    file_path = f'results/full_ensemble/{dataset_type}_ensemble_cif.h5'
    print(f"Loading {dataset_type} ensemble CIF from {file_path}")
    
    with h5py.File(file_path, 'r') as f:
        ensemble_cif = f['ensemble_cif'][:]
        n_models = f['ensemble_cif'].attrs.get('n_models', 24)
        method = f['ensemble_cif'].attrs.get('method', 'simple_average')
        
    print(f"  Shape: {ensemble_cif.shape}")
    print(f"  Number of models averaged: {n_models}")
    print(f"  Method: {method}")
    
    return ensemble_cif

def load_ground_truth_labels(dataset_type='temporal'):
    """
    Load ground truth duration and event labels.
    
    Args:
        dataset_type: 'temporal' or 'spatial'
    
    Returns:
        dict with 'duration' and 'event' arrays
    """
    file_path = f'results/final_deploy/{dataset_type}_test_labels.csv'
    print(f"Loading {dataset_type} ground truth labels from {file_path}")
    
    df = pd.read_csv(file_path)
    
    labels = {
        'duration': df['time'].values,
        'event': df['event'].values
    }
    
    print(f"  Number of samples: {len(labels['duration'])}")
    print(f"  Event distribution: {pd.Series(labels['event']).value_counts().sort_index().to_dict()}")
    
    return labels

def load_kfre_predictions(dataset_type='temporal'):
    """
    Load KFRE predictions from saved files.
    
    Args:
        dataset_type: 'temporal' or 'spatial'
    
    Returns:
        dict with KFRE predictions
    """
    file_path = f'results/kfre_predictions/{dataset_type}_kfre_predictions.csv'
    
    if not os.path.exists(file_path):
        print(f"KFRE predictions not found at {file_path}")
        print("Please run calculate_kfre_predictions.py first")
        return None
    
    print(f"Loading {dataset_type} KFRE predictions from {file_path}")
    df = pd.read_csv(file_path)
    
    kfre_predictions = {
        '4v2y': df['4v2y'].values,
        '4v5y': df['4v5y'].values,
        '8v2y': df['8v2y'].values,
        '8v5y': df['8v5y'].values
    }
    
    for key, values in kfre_predictions.items():
        print(f"  {key}: {np.sum(~np.isnan(values))} valid predictions")
    
    return kfre_predictions

def discretize_durations(durations, time_points=[365, 730, 1095, 1460, 1825]):
    """
    Discretize continuous durations to discrete time points.
    
    Args:
        durations: array of continuous durations
        time_points: list of discrete time points
    
    Returns:
        array of discretized durations
    """
    discretized = np.zeros_like(durations)
    
    for i, duration in enumerate(durations):
        # Find the closest time point
        closest_idx = np.argmin(np.abs(np.array(time_points) - duration))
        discretized[i] = time_points[closest_idx]
    
    return discretized

def fit_null_model(durations, events, time_points=[365, 730, 1095, 1460, 1825]):
    """
    Fit Aalen-Johansen model to get null model CIF.
    
    Args:
        durations: array of durations
        events: array of event indicators
        time_points: list of time points to evaluate
    
    Returns:
        dict with CIF values for each event at each time point
    """
    print("Fitting Aalen-Johansen null model...")
    
    # Create separate AJF for each event type
    null_cif = {}
    ajf_models = {}
    
    for event_type in [1, 2]:  # Dialysis and death
        ajf = AalenJohansenFitter()
        # Fit with specific event of interest
        ajf.fit(durations, events, event_of_interest=event_type)
        ajf_models[event_type] = ajf
        
        # Get CIF values at specified time points
        cif_values = []
        
        # Get the cumulative density dataframe
        cumulative_density = ajf.cumulative_density_
        
        # The column name for the event type might be a string
        # Check column names and find the right one
        col_name = None
        for col in cumulative_density.columns:
            if str(col) == str(event_type):
                col_name = col
                break
        
        if col_name is None:
            print(f"Warning: Could not find column for event type {event_type}")
            print(f"Available columns: {list(cumulative_density.columns)}")
            # Use zeros if column not found
            cif_values = [0.0] * len(time_points)
        else:
            for t in time_points:
                # Find the closest time point in the cumulative density index
                if t in cumulative_density.index:
                    cif_at_t = cumulative_density.loc[t, col_name]
                else:
                    # Interpolate if exact time point not available
                    # Find the last time before t
                    times_before = cumulative_density.index[cumulative_density.index <= t]
                    if len(times_before) > 0:
                        # Use the last available time before t
                        last_time = times_before[-1]
                        cif_at_t = cumulative_density.loc[last_time, col_name]
                    else:
                        # If t is before any observed time, use 0
                        cif_at_t = 0.0
                
                cif_values.append(float(cif_at_t))
        
        null_cif[f'event_{event_type}'] = np.array(cif_values)
    
    print(f"  Null model CIF calculated for {len(time_points)} time points")
    
    # Return the first event's AJF for compatibility
    return null_cif, ajf_models[1]

def calculate_metrics_with_bootstrap(y_true, y_pred, durations, events, n_bootstrap=50):
    """
    Calculate Brier score and C-index with bootstrap confidence intervals.
    
    Args:
        y_true: true binary outcomes at specific time
        y_pred: predicted probabilities
        durations: event times
        events: event indicators
        n_bootstrap: number of bootstrap iterations
    
    Returns:
        dict with metrics including mean and 95% CI
    """
    # Remove NaN predictions
    valid_mask = ~np.isnan(y_pred)
    
    if np.sum(valid_mask) == 0:
        return {
            'brier_score': {'mean': np.nan, 'lower': np.nan, 'upper': np.nan},
            'c_index': {'mean': np.nan, 'lower': np.nan, 'upper': np.nan},
            'n_valid': 0
        }
    
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    durations_valid = durations[valid_mask]
    events_valid = events[valid_mask]
    
    # Bootstrap
    brier_scores = []
    c_indices = []
    
    n_samples = len(y_true_valid)
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        idx = np.random.choice(n_samples, n_samples, replace=True)
        
        # Calculate metrics for this bootstrap sample
        try:
            brier = brier_score_loss(y_true_valid[idx], y_pred_valid[idx])
            c_index = concordance_index(durations_valid[idx], -y_pred_valid[idx], events_valid[idx])
            
            brier_scores.append(brier)
            c_indices.append(c_index)
        except:
            continue
    
    # Calculate statistics
    def get_stats(values):
        if len(values) == 0:
            return {'mean': np.nan, 'lower': np.nan, 'upper': np.nan}
        return {
            'mean': np.mean(values),
            'lower': np.percentile(values, 2.5),
            'upper': np.percentile(values, 97.5)
        }
    
    return {
        'brier_score': get_stats(brier_scores),
        'c_index': get_stats(c_indices),
        'n_valid': np.sum(valid_mask)
    }

def perform_quantile_analysis(predictions, observed_risk, n_quantiles=10):
    """
    Perform quantile-based analysis of predictions vs observed risk.
    
    Args:
        predictions: dict of model predictions
        observed_risk: observed risk values
        n_quantiles: number of quantiles
    
    Returns:
        dict with quantile analysis results
    """
    results = {}
    
    # For each model
    for model_name, model_preds in predictions.items():
        # Skip if all predictions are NaN
        if np.all(np.isnan(model_preds)):
            print(f"  Skipping {model_name} (all NaN)")
            continue
        
        # Remove NaN values for quantile calculation
        valid_mask = ~np.isnan(model_preds)
        valid_preds = model_preds[valid_mask]
        valid_observed = observed_risk[valid_mask]
        
        if len(valid_preds) < n_quantiles:
            print(f"  Skipping {model_name} (insufficient valid predictions)")
            continue
        
        # Calculate quantiles
        try:
            quantile_labels = pd.qcut(valid_preds, n_quantiles, labels=False, duplicates='drop')
            n_actual_quantiles = len(np.unique(quantile_labels))
        except:
            print(f"  Warning: Could not create {n_quantiles} quantiles for {model_name}")
            continue
        
        # Calculate mean predicted and observed risk in each quantile
        quantile_results = []
        
        for q in range(n_actual_quantiles):
            mask = quantile_labels == q
            
            quantile_data = {
                'quantile': q + 1,
                'n_samples': np.sum(mask),
                'predicted_risk': np.mean(valid_preds[mask]),
                'observed_risk': np.mean(valid_observed[mask]),
                'predicted_risk_std': np.std(valid_preds[mask]),
                'observed_risk_std': np.std(valid_observed[mask])
            }
            
            quantile_results.append(quantile_data)
        
        results[model_name] = quantile_results
    
    return results

def create_calibration_plot(quantile_results, metrics, time_point, dataset_type, output_dir):
    """
    Create bar chart calibration plot with metrics displayed.
    Uses consistent styling from metric_calculator.py
    
    Args:
        quantile_results: dict with quantile analysis results
        metrics: dict with model metrics including CI
        time_point: time point in days (730 or 1825)
        dataset_type: 'temporal' or 'spatial'
        output_dir: directory to save plots
    """
    # Create figure with consistent style from metric_calculator.py
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data for bar chart
    models = ['ensemble', 'kfre_4v', 'kfre_8v']
    model_labels = ['Ensemble', 'KFRE 4v', 'KFRE 8v']
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e']  # Blue, green, orange
    
    # Get number of quantiles
    n_quantiles = len(list(quantile_results.values())[0]) if quantile_results else 10
    x = np.arange(1, n_quantiles + 1)
    width = 0.25
    
    # Plot bars for each model
    for i, (model, label, color) in enumerate(zip(models, model_labels, colors)):
        if model in quantile_results:
            quantile_data = quantile_results[model]
            predicted_risks = [q['predicted_risk'] * 100 for q in quantile_data]
            
            offset = (i - 1) * width
            bars = ax.bar(x + offset, predicted_risks, width, label=label, color=color, alpha=0.8)
            
            # Add value labels on bars (only if risk > 1%)
            for j, (bar, risk) in enumerate(zip(bars, predicted_risks)):
                if risk > 1:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           f'{risk:.1f}', ha='center', va='bottom', fontsize=8)
    
    # Plot observed risk as a line with markers
    if 'ensemble' in quantile_results:  # Use ensemble quantiles for observed risk
        quantile_data = quantile_results['ensemble']
        observed_risks = [q['observed_risk'] * 100 for q in quantile_data]
        ax.plot(x, observed_risks, 'k-', linewidth=2.5, marker='o', markersize=8,
                label='Observed Risk', zorder=5)
        
        # Add value labels for observed risk (only if risk > 1%)
        for j, risk in enumerate(observed_risks):
            if risk > 1:
                ax.text(x[j], risk + 1, f'{risk:.1f}', ha='center', va='bottom',
                       fontsize=8, fontweight='bold')
    
    # Formatting
    ax.set_xlabel('Risk Quantile', fontsize=12)
    ax.set_ylabel('Risk (%)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Q{i}' for i in range(1, n_quantiles + 1)])
    
    # Set y-axis limit based on data
    max_risk = max([max([q['predicted_risk'] * 100 for q in quantile_results[m]])
                    for m in models if m in quantile_results] +
                   [max([q['observed_risk'] * 100 for q in quantile_results['ensemble']])])
    ax.set_ylim(0, min(100, max_risk * 1.2))
    
    # Create legend first
    legend = ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add metrics text box next to legend
    metrics_text = []
    for model, label in zip(models, model_labels):
        if model in metrics:
            m = metrics[model]
            metrics_text.append(f"{label}:")
            # Add Brier score
            if 'brier_score' in m:
                metrics_text.append(f"  Brier: {m['brier_score']['mean']:.3f} ({m['brier_score']['lower']:.3f}-{m['brier_score']['upper']:.3f})")
            # Add C-index
            if 'c_index' in m:
                metrics_text.append(f"  C-index: {m['c_index']['mean']:.3f} ({m['c_index']['lower']:.3f}-{m['c_index']['upper']:.3f})")
            # Add IPA
            if 'ipa' in m and m['ipa']['mean'] is not None:
                metrics_text.append(f"  IPA: {m['ipa']['mean']:.3f} ({m['ipa']['lower']:.3f}-{m['ipa']['upper']:.3f})")
            metrics_text.append("")  # Empty line between models
    
    if metrics_text:
        # Remove last empty line
        if metrics_text[-1] == "":
            metrics_text = metrics_text[:-1]
        
        # Create text box positioned next to legend
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        metrics_str = '\n'.join(metrics_text)
        
        # Get legend position and size
        legend_bbox = legend.get_window_extent()
        ax_bbox = ax.get_window_extent()
        
        # Position metrics box to the right of the legend
        # Using relative coordinates
        x_position = 0.02 + (legend_bbox.width / ax_bbox.width) + 0.02  # Legend x + legend width + padding
        y_position = 0.98  # Same top position as legend
        
        ax.text(x_position, y_position, metrics_str, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='left', bbox=props)
    
    # Add title
    year_label = f"{time_point//365} Year" if time_point//365 == 1 else f"{time_point//365} Years"
    ax.set_title(f'Calibration Plot - {dataset_type.capitalize()} Dataset ({year_label})',
                 fontsize=14, pad=10)
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f'{dataset_type}_{time_point}days_calibration.png'
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved calibration plot to {plot_path}")
    
    return plot_path


def create_dca_plot(predictions, labels, time_point, dataset_type, output_dir):
    """
    Create decision curve analysis plot comparing models.
    Uses the same style as src/dca.py
    
    Args:
        predictions: dict with model predictions
        labels: dict with duration and event arrays
        time_point: time point in days (730 or 1825)
        dataset_type: 'temporal' or 'spatial'
        output_dir: directory to save plots
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define threshold range
    thresholds = np.linspace(0.01, 0.5, 50)
    
    # Colors for different models
    colors = {
        'ensemble': '#1f77b4',  # Blue
        'kfre_4v': '#2ca02c',   # Green
        'kfre_8v': '#ff7f0e'    # Orange
    }
    
    # Calculate decision curves for each model
    for model_name, model_pred in predictions.items():
        # Remove NaN values
        valid_mask = ~np.isnan(model_pred)
        if np.sum(valid_mask) == 0:
            continue
            
        # Get valid predictions and corresponding labels
        risk_valid = model_pred[valid_mask]
        durations_valid = labels['duration'][valid_mask]
        events_valid = labels['event'][valid_mask]
        
        # Calculate decision curve
        dc_result = decision_curve(
            risk=risk_valid,
            durations=durations_valid,
            events=events_valid,
            horizon=time_point,
            thresholds=thresholds,
            ipcw=True
        )
        
        # Plot model curve
        label = {
            'ensemble': 'Ensemble',
            'kfre_4v': 'KFRE 4v',
            'kfre_8v': 'KFRE 8v'
        }[model_name]
        
        ax.plot(thresholds, dc_result['nb_model'], '-',
                linewidth=2.5, color=colors[model_name], label=label)
    
    # Plot treat all and treat none (calculate once using ensemble data)
    if 'ensemble' in predictions:
        valid_mask = ~np.isnan(predictions['ensemble'])
        durations_valid = labels['duration'][valid_mask]
        events_valid = labels['event'][valid_mask]
        
        # Calculate prevalence for treat all curve
        event_at_horizon = (events_valid == 1) & (durations_valid <= time_point)
        prevalence = np.mean(event_at_horizon)
        
        # Calculate treat all and treat none curves
        nb_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
        nb_none = np.zeros_like(thresholds)
        
        # Plot reference curves
        ax.plot(thresholds, nb_all, '--', linewidth=1.5,
                color='gray', label="Treat All")
        ax.plot(thresholds, nb_none, '-', linewidth=1.5,
                color='black', label="Treat None")
    
    # Formatting (matching dca.py style)
    ax.set_xlabel("Risk Threshold", fontsize=12)
    ax.set_ylabel("Net Benefit", fontsize=12)
    
    # Title
    year_label = f"{time_point//365} Year" if time_point//365 == 1 else f"{time_point//365} Years"
    ax.set_title(f"Decision Curve Analysis - {dataset_type.capitalize()} Dataset ({year_label})",
                 fontsize=14)
    
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set y-axis lower limit to 0 to hide negative net benefit values
    ax.set_ylim(bottom=0)
    
    # Set x-axis to show percentages
    ax.set_xlim(0, 0.5)
    ax.set_xticks(np.arange(0, 0.6, 0.1))
    ax.set_xticklabels([f"{int(x*100)}%" for x in np.arange(0, 0.6, 0.1)])
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f'{dataset_type}_{time_point}days_dca.png'
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved DCA plot to {plot_path}")
    
    return plot_path

def main():
    """
    Main analysis function.
    """
    print("="*80)
    print("Ensemble vs KFRE vs Null Model Comparison")
    print("="*80)
    
    # Create output directory
    output_dir = 'results/ensemble_kfre_comparison'
    os.makedirs(output_dir, exist_ok=True)
    
    # Time points of interest
    time_points = [365, 730, 1095, 1460, 1825]
    analysis_time_points = [730, 1825]  # 2 and 5 years
    
    # Store all results
    all_results = {}
    
    # Process each dataset
    for dataset_type in ['temporal', 'spatial']:
        print(f"\n{'='*60}")
        print(f"Processing {dataset_type.upper()} dataset")
        print(f"{'='*60}")
        
        # Load data
        ensemble_cif = load_ensemble_cif(dataset_type)
        labels = load_ground_truth_labels(dataset_type)
        kfre_preds = load_kfre_predictions(dataset_type)
        
        if kfre_preds is None:
            print(f"Skipping {dataset_type} dataset due to missing KFRE predictions")
            continue
        
        # Fit null model
        null_cif, ajf = fit_null_model(labels['duration'], labels['event'], time_points)
        
        # Focus on dialysis event (Event 1)
        event_idx = 0  # First event in CIF array
        
        dataset_results = {}
        
        # Analyze each time point
        for t_idx, t_days in enumerate(analysis_time_points):
            print(f"\n--- Analyzing {t_days} days ({t_days/365:.0f} years) ---")
            
            # Get time point index in the full array
            full_t_idx = time_points.index(t_days)
            
            # Extract predictions for this time point
            ensemble_pred = ensemble_cif[event_idx, full_t_idx, :]
            
            # Get KFRE predictions for this time point
            if t_days == 730:  # 2 years
                kfre_4v_pred = kfre_preds['4v2y']
                kfre_8v_pred = kfre_preds['8v2y']
            else:  # 5 years (1825 days)
                kfre_4v_pred = kfre_preds['4v5y']
                kfre_8v_pred = kfre_preds['8v5y']
            
            # Calculate observed risk at this time point
            observed_risk = (labels['duration'] <= t_days) & (labels['event'] == 1)
            
            # Prepare predictions dict
            predictions = {
                'ensemble': ensemble_pred,
                'kfre_4v': kfre_4v_pred,
                'kfre_8v': kfre_8v_pred
            }
            
            # Calculate metrics with bootstrap for each model
            print("\n  Calculating metrics with bootstrap (50 iterations)...")
            metrics = {}
            
            for model_name, model_pred in predictions.items():
                print(f"    Processing {model_name}...")
                metrics[model_name] = calculate_metrics_with_bootstrap(
                    observed_risk, model_pred,
                    labels['duration'], labels['event'],
                    n_bootstrap=50
                )
            
            # Calculate null model metrics
            print("    Processing null model...")
            null_pred = np.full_like(observed_risk, null_cif['event_1'][full_t_idx], dtype=float)
            null_metrics = calculate_metrics_with_bootstrap(
                observed_risk, null_pred,
                labels['duration'], labels['event'],
                n_bootstrap=50
            )
            metrics['null'] = null_metrics
            
            # Calculate IPA (Index of Prediction Accuracy) with CI
            for model_name in ['ensemble', 'kfre_4v', 'kfre_8v']:
                model_brier = metrics[model_name]['brier_score']
                null_brier = metrics['null']['brier_score']
                
                if not np.isnan(model_brier['mean']) and not np.isnan(null_brier['mean']):
                    # Calculate IPA for mean and CI bounds
                    ipa_mean = 1 - model_brier['mean'] / null_brier['mean']
                    # For CI, use conservative approach
                    ipa_lower = 1 - model_brier['upper'] / null_brier['lower']
                    ipa_upper = 1 - model_brier['lower'] / null_brier['upper']
                    
                    metrics[model_name]['ipa'] = {
                        'mean': ipa_mean,
                        'lower': ipa_lower,
                        'upper': ipa_upper
                    }
                else:
                    metrics[model_name]['ipa'] = {
                        'mean': np.nan,
                        'lower': np.nan,
                        'upper': np.nan
                    }
            
            # Print metrics summary
            print("\n  Metrics Summary:")
            for model_name in ['ensemble', 'kfre_4v', 'kfre_8v']:
                m = metrics[model_name]
                print(f"    {model_name}:")
                print(f"      Brier: {m['brier_score']['mean']:.4f} ({m['brier_score']['lower']:.4f}-{m['brier_score']['upper']:.4f})")
                print(f"      C-index: {m['c_index']['mean']:.4f} ({m['c_index']['lower']:.4f}-{m['c_index']['upper']:.4f})")
                if 'ipa' in m:
                    print(f"      IPA: {m['ipa']['mean']:.4f} ({m['ipa']['lower']:.4f}-{m['ipa']['upper']:.4f})")
            
            # Perform quantile analysis
            print("\n  Performing quantile analysis...")
            
            # Calculate observed risk in quantiles based on ensemble predictions
            quantile_results = perform_quantile_analysis(predictions, observed_risk.astype(float))
            
            # Create calibration plot with metrics
            create_calibration_plot(quantile_results, metrics, t_days, dataset_type, output_dir)
            
            # Create decision curve analysis plot
            print("\n  Creating decision curve analysis...")
            create_dca_plot(predictions, labels, t_days, dataset_type, output_dir)
            
            # Create decision curve analysis plot
            print("\n  Creating decision curve analysis...")
            create_dca_plot(predictions, labels, t_days, dataset_type, output_dir)
            
            # Store results
            dataset_results[f'{t_days}days'] = {
                'metrics': metrics,
                'quantile_analysis': quantile_results
            }
        
        all_results[dataset_type] = dataset_results
    
    # Save all results to JSON
    results_path = os.path.join(output_dir, 'comparison_results.json')
    
    # Convert numpy values to Python native types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_to_serializable(all_results)
    
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': serializable_results
        }, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Analysis complete! Results saved to {output_dir}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()