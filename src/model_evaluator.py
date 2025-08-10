"""Model evaluation module for CKD Risk Prediction model
Containing code to calculate and plot the integrated concordance index, brier score, integrated brier score, etc. for null model and trained model predictions"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from typing import Dict, List, Union, Optional, Tuple
from scipy import stats
from pathlib import Path

# Import utility functions
from src.util import observed_risk_calculator

# Set up logging
logger = logging.getLogger(__name__)

class NullModel:
    """Class for calculating null model metrics"""
    
    @staticmethod
    def calculate_null_brier(durations, events, time_grid, event_of_interest=1):
        """
        Calculate the null model Brier score for a given set of durations and events.

        Args:
            durations (array-like): Event/censoring times.
            events (array-like): Event indicators (1 for event, 0 for censoring).
            time_grid (list): List of time points for evaluation.
            event_of_interest (int): The event type of interest for competing risks.

        Returns:
            float: Integrated Brier score for the null model.
        """
        try:
            from lifelines import KaplanMeierFitter
            from pycox.evaluation import EvalSurv
            
            def interpolate_cif(cif, time_points):
                cif_values = cif.values.squeeze()
                cif_index = cif.index.values
                max_time = cif_index[-1]
                max_cif_value = cif_values[-1]
                
                # Interpolation within range
                interpolated = np.interp(time_points, cif_index, cif_values)
                
                # Handle extrapolation for times > max_time
                interpolated[time_points > max_time] = max_cif_value
                
                return interpolated
                
            # Ensure durations and events are NumPy arrays with numeric values
            try:
                # Convert to numeric, coercing non-numeric values to NaN
                durations = pd.to_numeric(durations, errors='coerce')
                events = pd.to_numeric(events, errors='coerce')
                
                # Convert to numpy arrays
                durations = np.array(durations).squeeze()
                events = np.array(events).squeeze()
                
                # Filter out NaN values
                valid_mask = ~np.isnan(durations) & ~np.isnan(events)
                if not np.all(valid_mask):
                    logger.warning(f"Removing {(~valid_mask).sum()} NaN values from durations/events")
                    durations = durations[valid_mask]
                    events = events[valid_mask]
                    
                # If no valid data after filtering, return NaN
                if len(durations) == 0:
                    logger.error("No valid numeric data for Brier score calculation")
                    return np.nan
            except Exception as e:
                logger.error(f"Error converting duration/event data to numeric: {e}")
                return np.nan
            
            # Fit Kaplan-Meier estimator
            kmf = KaplanMeierFitter()
            kmf.fit(durations, events)
            cif = kmf.cumulative_density_
            
            # Calculate survival probabilities
            surv_probs = 1 - interpolate_cif(cif, time_grid)
            
            # Create survival probability DataFrame for EvalSurv
            surv_df = pd.DataFrame(
                np.tile(surv_probs, (len(durations), 1)).T,
                index=time_grid
            )

            # Evaluate using EvalSurv
            ev = EvalSurv(surv_df, durations, events == event_of_interest, censor_surv="km")

            # Integrated Brier score
            brier = ev.brier_score(time_grid)
            return brier
            
        except Exception as e:
            logger.error(f"Error calculating null Brier score: {e}")
            return np.nan

class RiskComparer:
    """Module to compare predicted risk from different models to observed risk in a dataframe then generate plot"""
    
    @staticmethod
    def timepoint_risk_comparison(df: pd.DataFrame,
                                 timepoints: List[float],
                                 n_bootstrap: int,
                                 n_quantile: int,
                                 output_path: str,
                                 risk_cols: List[str],
                                 duration_col: str,
                                 event_col: str,
                                 baseline_col: str = 'observed_risk') -> Dict:
        """
        Function to compare predicted and observed risk, grouped to quantiles and produce plot.
        
        Args:
            df: DataFrame containing patient data
            timepoints: List of time points to evaluate at
            n_bootstrap: Number of bootstrap iterations
            n_quantile: Number of quantiles to divide predictions into
            output_path: Path to save output plots and results
            risk_cols: List of column names containing predicted risks
            duration_col: Column name for duration/time
            event_col: Column name for event indicator
            baseline_col: Name for the observed risk column
            
        Returns:
            Dictionary containing evaluation metrics
        """
        try:
            from sklearn.metrics import brier_score_loss
            from lifelines.utils import concordance_index
            from scipy.stats import chi2
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Create output directory if it doesn't exist
            os.makedirs(output_path, exist_ok=True)
            
            # Initialize results
            bootstrap_results = []
            
            # For each bootstrap iteration
            for i in range(n_bootstrap):
                logger.info(f"Running bootstrap iteration {i+1}/{n_bootstrap}")
                
                # Sample with replacement
                bootstrap_indices = np.random.choice(df.index, size=len(df), replace=True)
                df_bootstrap = df.loc[bootstrap_indices].copy()
                
                # Calculate null Brier scores for this bootstrap sample
                null_brier_scores = {}
                for time in timepoints:
                    try:
                        # Convert to numeric values
                        durations = pd.to_numeric(df_bootstrap[duration_col], errors='coerce')
                        events = pd.to_numeric(df_bootstrap[event_col], errors='coerce')
                        
                        # Remove NaN values
                        valid_mask = ~np.isnan(durations) & ~np.isnan(events)
                        if not np.all(valid_mask):
                            logger.warning(f"Removing {(~valid_mask).sum()} NaN values from durations/events")
                            durations = durations[valid_mask]
                            events = events[valid_mask]
                        
                        # Check if we have enough data
                        if len(durations) == 0:
                            logger.warning("No valid data for null Brier score calculation")
                            null_brier_scores[time] = np.nan
                            continue
                        
                        null_brier = NullModel.calculate_null_brier(
                            durations,
                            events,
                            [time]
                        )
                        null_brier_scores[time] = null_brier
                    except Exception as e:
                        logger.error(f"Error calculating null Brier score: {e}")
                        null_brier_scores[time] = np.nan
                
                # For each risk column
                for risk_col in risk_cols:
                    logger.info(f"Processing risk column: {risk_col}")
                    
                    # For each time point
                    for time in timepoints:
                        logger.info(f"Processing time point: {time}")
                        
                        # Create a copy for this analysis
                        analysis_df = df_bootstrap.copy()
                        
                        # Divide into quantiles
                        analysis_df['quantile'] = pd.qcut(
                            analysis_df[risk_col],
                            n_quantile,
                            labels=False,
                            duplicates='drop'
                        )
                        
                        # Calculate observed risk for each quantile
                        quantile_results = []
                        
                        for q in analysis_df['quantile'].unique():
                            # Get subset for this quantile
                            quantile_df = analysis_df[analysis_df['quantile'] == q].copy()
                            
                            # Calculate observed risk
                            quantile_df = observed_risk_calculator(
                                quantile_df,
                                duration=duration_col,
                                event=event_col,
                                time_points=[time],
                                output_col=f"{baseline_col}_{time}"
                            )
                            
                            # Store results
                            # Check if the observed risk column was created successfully
                            observed_risk_col = f"{baseline_col}_{time}"
                            if observed_risk_col in quantile_df.columns:
                                observed_risk = quantile_df[observed_risk_col].mean()
                                predicted_risk = quantile_df[risk_col].mean()
                                
                                quantile_results.append({
                                    'quantile': q,
                                    'observed_risk': observed_risk,
                                    'predicted_risk': predicted_risk,
                                    'n': len(quantile_df)
                                })
                            else:
                                logger.warning(f"Observed risk column '{observed_risk_col}' not found in dataframe")
                                # Skip this quantile
                                continue
                        
                        # Convert to DataFrame
                        quantile_df = pd.DataFrame(quantile_results)
                        
                        # Check if we have any results
                        if len(quantile_df) == 0:
                            logger.warning(f"No valid quantiles for risk column {risk_col} at time {time}")
                            continue
                        
                        # Calculate metrics
                        try:
                            # Chi-square goodness of fit
                            chi2_stat = np.sum(
                                ((quantile_df['observed_risk'] - quantile_df['predicted_risk']) ** 2) /
                                (quantile_df['predicted_risk'] * (1 - quantile_df['predicted_risk'] / quantile_df['n']))
                            )
                            dof = len(quantile_df) - 2
                            p_value = 1 - chi2.cdf(chi2_stat, dof)
                        except Exception as e:
                            logger.warning(f"Error calculating chi-square goodness of fit: {e}")
                            chi2_stat = np.nan
                            p_value = np.nan
                        
                        # Brier score
                        try:
                            # Convert to numeric and handle non-numeric values
                            durations = pd.to_numeric(df_bootstrap[duration_col], errors='coerce')
                            events = pd.to_numeric(df_bootstrap[event_col], errors='coerce')
                            risks = pd.to_numeric(df_bootstrap[risk_col], errors='coerce')
                            
                            # Create mask for valid rows
                            valid_mask = ~np.isnan(durations) & ~np.isnan(events) & ~np.isnan(risks)
                            
                            if valid_mask.sum() > 0:
                                brier_score = brier_score_loss(
                                    (durations[valid_mask] <= time) & (events[valid_mask] == 1),
                                    risks[valid_mask]
                                )
                            else:
                                logger.warning("No valid data for Brier score calculation")
                                brier_score = np.nan
                        except Exception as e:
                            logger.warning(f"Error calculating Brier score: {e}")
                            brier_score = np.nan
                        
                        # C-index
                        try:
                            # Use the same valid mask from Brier score calculation
                            if valid_mask.sum() > 0:
                                c_index = concordance_index(
                                    durations[valid_mask],
                                    -risks[valid_mask],
                                    events[valid_mask]
                                )
                            else:
                                logger.warning("No valid data for C-index calculation")
                                c_index = np.nan
                        except Exception as e:
                            logger.warning(f"Error calculating C-index: {e}")
                            c_index = np.nan
                        
                        # Store bootstrap result
                        bootstrap_results.append({
                            'bootstrap': i,
                            'time': time,
                            'risk_col': risk_col,
                            'brier_score': brier_score,
                            'c_index': c_index,
                            'chi2_stat': chi2_stat,
                            'p_value': p_value,
                            'null_brier_score': null_brier_scores[time]
                        })
                        
                        # Create calibration plot for this time point and risk column
                        plt.figure(figsize=(10, 8))
                        
                        # Plot identity line
                        plt.plot([0, 1], [0, 1], 'k--', label='Ideal')
                        
                        # Plot calibration points
                        plt.scatter(
                            quantile_df['predicted_risk'],
                            quantile_df['observed_risk'],
                            s=100 * quantile_df['n'] / quantile_df['n'].max(),
                            alpha=0.7
                        )
                        
                        # Add error bars (95% CI)
                        for _, row in quantile_df.iterrows():
                            # Calculate Wilson score interval
                            n = row['n']
                            p = row['observed_risk']
                            z = 1.96  # 95% CI
                            
                            # Wilson score interval
                            denominator = 1 + z**2/n
                            center = (p + z**2/(2*n)) / denominator
                            halfwidth = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denominator
                            
                            lower = max(0, center - halfwidth)
                            upper = min(1, center + halfwidth)
                            
                            plt.plot(
                                [row['predicted_risk'], row['predicted_risk']],
                                [lower, upper],
                                'b-'
                            )
                        
                        # Add labels and title
                        plt.xlabel('Predicted Risk')
                        plt.ylabel('Observed Risk')
                        plt.title(f'Calibration Plot - {risk_col} at {time} years (Bootstrap {i+1})')
                        
                        # Add metrics as text
                        plt.text(
                            0.05, 0.95,
                            f'C-index: {c_index:.3f}\n'
                            f'Brier Score: {brier_score:.3f}\n'
                            f'Chi-square: {chi2_stat:.3f} (p={p_value:.3f})',
                            transform=plt.gca().transAxes,
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
                        )
                        
                        # Save plot
                        plot_filename = f"{risk_col}_time{time}_bootstrap{i+1}.png"
                        plt.savefig(os.path.join(output_path, plot_filename), dpi=300, bbox_inches='tight')
                        plt.close()
            
            # Convert results to DataFrame
            results_df = pd.DataFrame(bootstrap_results)
            
            # Save results to CSV
            results_df.to_csv(os.path.join(output_path, "bootstrap_results.csv"), index=False)
            
            # Create summary plots
            # Boxplot of C-index by risk column and time
            plt.figure(figsize=(12, 8))
            sns.boxplot(x='risk_col', y='c_index', hue='time', data=results_df)
            plt.title('C-index by Risk Column and Time')
            plt.savefig(os.path.join(output_path, "c_index_summary.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Boxplot of Brier score by risk column and time
            plt.figure(figsize=(12, 8))
            sns.boxplot(x='risk_col', y='brier_score', hue='time', data=results_df)
            plt.title('Brier Score by Risk Column and Time')
            plt.savefig(os.path.join(output_path, "brier_score_summary.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Calculate summary statistics
            summary = results_df.groupby(['risk_col', 'time']).agg({
                'c_index': ['mean', 'std', 'min', 'max'],
                'brier_score': ['mean', 'std', 'min', 'max'],
                'null_brier_score': ['mean'],
                'chi2_stat': ['mean'],
                'p_value': ['mean']
            })
            
            # Calculate relative improvement over null model
            summary['brier_improvement'] = 1 - summary[('brier_score', 'mean')] / summary[('null_brier_score', 'mean')]
            
            # Save summary to CSV
            summary.to_csv(os.path.join(output_path, "summary_statistics.csv"))
            
            logger.info(f"Risk comparison completed. Results saved to {output_path}")
            
            # Convert the MultiIndex DataFrame to a dictionary with string keys
            # This avoids the "keys must be str, int, float, bool or None, not tuple" error
            # when serializing to JSON
            result_dict = {}
            for (risk_col, time), row in summary.iterrows():
                key = f"{risk_col}_{time}"
                result_dict[key] = row.to_dict()
            
            return result_dict
            
        except Exception as e:
            logger.error(f"Error in timepoint_risk_comparison: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}