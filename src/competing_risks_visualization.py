#!/usr/bin/env python3
"""
Visualization functions for DeepHit competing risks evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import os
from datetime import datetime

# Import existing visualization functions
from src.metric_calculator import plot_calibration, plot_metrics_by_time
from src.dca import plot_decision_curves_subplots
from src.dca import decision_curve, risk_at_horizon


def extract_event_specific_predictions(
    predictions: pd.DataFrame,
    time_grid: np.ndarray,
    event: int = 1
) -> pd.DataFrame:
    """
    Extract event-specific predictions from DeepHit competing risks format.
    
    Args:
        predictions: DataFrame with competing risks predictions (10, n_samples)
        time_grid: Array of time points [365, 730, 1095, 1460, 1825]
        event: Event number (1 or 2)
        
    Returns:
        DataFrame with event-specific predictions (5, n_samples)
    """
    num_time_points = len(time_grid)
    
    if event == 1:
        # Event 1: rows 0-4
        event_predictions = predictions.iloc[0:num_time_points, :]
    elif event == 2:
        # Event 2: rows 5-9
        event_predictions = predictions.iloc[num_time_points:2*num_time_points, :]
    else:
        raise ValueError(f"Event must be 1 or 2, got {event}")
    
    # Reset index to time grid
    event_predictions.index = time_grid
    
    return event_predictions


def create_competing_risks_visualizations(
    predictions: pd.DataFrame,
    durations: np.ndarray,
    events: np.ndarray,
    time_horizons: List[int],
    output_dir: str,
    dataset_name: str,
    model_type: str,
    hp_config: Dict[str, Any],
    timestamp: str = None
) -> Dict[str, Any]:
    """
    Create separate visualizations for each competing event in DeepHit.
    
    Args:
        predictions: DataFrame with competing risks predictions
        durations: Array of event times
        events: Array of event indicators (0, 1, 2)
        time_horizons: List of time horizons for evaluation
        output_dir: Directory to save plots
        dataset_name: Name of dataset (e.g., 'temporal_test', 'spatial_test')
        model_type: Type of model ('deephit')
        hp_config: Hyperparameter configuration
        timestamp: Timestamp for file naming
        
    Returns:
        Dictionary with visualization paths for each event
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Check if this is competing risks format
    time_grid = np.array([365, 730, 1095, 1460, 1825])
    expected_rows = len(time_grid) * 2  # 2 causes × 5 time points = 10 rows
    
    if predictions.shape[0] != expected_rows:
        print(f"Warning: Expected {expected_rows} rows for competing risks, got {predictions.shape[0]}")
        print("Falling back to standard visualization")
        return {}
    
    print(f"Creating competing risks visualizations for {dataset_name} dataset")
    
    visualization_paths = {}
    
    # Event names for labeling
    event_names = {
        1: "Event_1_RRT_eGFR15",
        2: "Event_2_Mortality"
    }
    
    event_labels = {
        1: "Event 1 (RRT/eGFR<15)",
        2: "Event 2 (Mortality)"
    }
    
    # Process each competing event
    for event_num in [1, 2]:
        print(f"\n--- Creating visualizations for {event_labels[event_num]} ---")
        
        # Extract event-specific predictions
        event_predictions = extract_event_specific_predictions(
            predictions, time_grid, event_num
        )
        
        # Create binary events for this specific event
        binary_events = (events == event_num).astype(int)
        event_rate = binary_events.mean()
        
        print(f"{event_labels[event_num]} rate: {event_rate:.3f}")
        print(f"Event-specific predictions shape: {event_predictions.shape}")
        
        # Skip if no events of this type
        if event_rate == 0:
            print(f"No events of type {event_num}, skipping visualizations")
            continue
        
        event_name = event_names[event_num]
        
        # 1. Calibration plot
        calibration_path = os.path.join(
            output_dir, f"{dataset_name}_{event_name}_calibration_{timestamp}.png"
        )
        try:
            plot_calibration(
                event_predictions, durations, binary_events, 
                time_horizons, calibration_path, model_type
            )
            visualization_paths[f'{event_name}_calibration'] = calibration_path
            print(f"Created calibration plot: {calibration_path}")
        except Exception as e:
            print(f"Error creating calibration plot for {event_name}: {e}")
        
        # 2. Decision Curve Analysis
        dca_cfg = hp_config.get("evaluation", {}).get("dca", {})
        if dca_cfg.get("enable", False):
            print(f"Creating DCA plots for {event_labels[event_num]}")
            
            # Get DCA configuration
            horizons = dca_cfg.get("horizons", [365])
            thr_grid = np.linspace(
                dca_cfg.get("threshold_grid", {}).get("start", 0.01),
                dca_cfg.get("threshold_grid", {}).get("stop", 0.50),
                dca_cfg.get("threshold_grid", {}).get("num", 50),
            )
            use_ipcw = dca_cfg.get("ipcw", True)
            
            dca_subplot_data = []
            
            for H in horizons:
                try:
                    # Convert survival → risk for this event
                    risk_H = risk_at_horizon(event_predictions, horizon=H)
                    
                    # Compute DCA components
                    nb = decision_curve(
                        risk_H,
                        durations,
                        binary_events,
                        horizon=H,
                        thresholds=thr_grid,
                        ipcw=use_ipcw
                    )
                    
                    # Collect data for subplot
                    dca_subplot_data.append({
                        'thresholds': nb["thresholds"],
                        'net_benefit': nb["nb_model"],
                        'net_benefit_all': nb["nb_treat_all"],
                        'net_benefit_none': nb["nb_treat_none"],
                        'horizon': H,
                        'title': f'{event_labels[event_num]} - {H} days',
                        'label': f'DeepHit Model'
                    })
                    
                    print(f"Computed DCA for {event_labels[event_num]} at {H} days")
                    
                except Exception as e:
                    print(f"Error computing DCA for {event_name} at {H} days: {e}")
                    continue
            
            # Create DCA subplot
            if dca_subplot_data:
                dca_path = os.path.join(
                    output_dir, f"{dataset_name}_{event_name}_dca_subplots_{timestamp}.png"
                )
                try:
                    plot_decision_curves_subplots(dca_subplot_data, dca_path)
                    visualization_paths[f'{event_name}_dca_subplots'] = dca_path
                    print(f"Created DCA plot: {dca_path}")
                except Exception as e:
                    print(f"Error creating DCA plot for {event_name}: {e}")
    
    # 3. Combined comparison plot
    try:
        combined_path = os.path.join(
            output_dir, f"{dataset_name}_competing_risks_comparison_{timestamp}.png"
        )
        create_competing_risks_comparison_plot(
            predictions, durations, events, time_grid, combined_path
        )
        visualization_paths['competing_risks_comparison'] = combined_path
        print(f"Created competing risks comparison plot: {combined_path}")
    except Exception as e:
        print(f"Error creating competing risks comparison plot: {e}")
    
    return visualization_paths


def create_competing_risks_comparison_plot(
    predictions: pd.DataFrame,
    durations: np.ndarray,
    events: np.ndarray,
    time_grid: np.ndarray,
    output_path: str
) -> None:
    """
    Create a comparison plot showing both competing events.
    
    Args:
        predictions: DataFrame with competing risks predictions
        durations: Array of event times
        events: Array of event indicators
        time_grid: Array of time points
        output_path: Path to save the plot
    """
    # Extract predictions for both events
    event1_preds = extract_event_specific_predictions(predictions, time_grid, 1)
    event2_preds = extract_event_specific_predictions(predictions, time_grid, 2)
    
    # Calculate mean risk over time for each event
    event1_mean_risk = 1 - event1_preds.mean(axis=1)  # Convert survival to risk
    event2_mean_risk = 1 - event2_preds.mean(axis=1)
    
    # Calculate observed event rates
    event1_rate = (events == 1).mean()
    event2_rate = (events == 2).mean()
    censoring_rate = (events == 0).mean()
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Mean predicted risks over time
    ax1.plot(time_grid, event1_mean_risk, 'b-', linewidth=2, label='Event 1 (RRT/eGFR<15)')
    ax1.plot(time_grid, event2_mean_risk, 'r-', linewidth=2, label='Event 2 (Mortality)')
    ax1.set_xlabel('Time (days)')
    ax1.set_ylabel('Mean Predicted Risk')
    ax1.set_title('Mean Predicted Risks Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Event distribution
    event_counts = [censoring_rate, event1_rate, event2_rate]
    event_labels = ['Censored', 'Event 1\n(RRT/eGFR<15)', 'Event 2\n(Mortality)']
    colors = ['gray', 'blue', 'red']
    
    bars = ax2.bar(event_labels, event_counts, color=colors, alpha=0.7)
    ax2.set_ylabel('Proportion')
    ax2.set_title('Event Distribution in Dataset')
    ax2.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, count in zip(bars, event_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{count:.3f}', ha='center', va='bottom')
    
    # Plot 3: Risk distribution at final time point
    final_time_idx = -1  # Last time point
    event1_final_risks = 1 - event1_preds.iloc[final_time_idx, :]
    event2_final_risks = 1 - event2_preds.iloc[final_time_idx, :]
    
    ax3.hist(event1_final_risks, bins=30, alpha=0.6, label='Event 1 (RRT/eGFR<15)', color='blue')
    ax3.hist(event2_final_risks, bins=30, alpha=0.6, label='Event 2 (Mortality)', color='red')
    ax3.set_xlabel(f'Predicted Risk at {time_grid[final_time_idx]} days')
    ax3.set_ylabel('Frequency')
    ax3.set_title(f'Risk Distribution at {time_grid[final_time_idx]} days')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Correlation between competing risks
    correlation = np.corrcoef(event1_final_risks, event2_final_risks)[0, 1]
    
    ax4.scatter(event1_final_risks, event2_final_risks, alpha=0.5, s=1)
    ax4.set_xlabel('Event 1 Risk (RRT/eGFR<15)')
    ax4.set_ylabel('Event 2 Risk (Mortality)')
    ax4.set_title(f'Competing Risks Correlation (r={correlation:.3f})')
    ax4.grid(True, alpha=0.3)
    
    # Add diagonal line
    max_risk = max(event1_final_risks.max(), event2_final_risks.max())
    ax4.plot([0, max_risk], [0, max_risk], 'k--', alpha=0.5, label='Equal risk line')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved competing risks comparison plot to {output_path}")