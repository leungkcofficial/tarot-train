"""Model evaluation step for CKD Risk Prediction Models
Containing code to evaluate KFRE model performance"""

import pandas as pd
import numpy as np
import os
from zenml.steps import step
from typing import Dict, Optional, List
from src.KFRE import KFRECalculator
from src.util import df_event_focus
from src.model_evaluator import NullModel
from lifelines import KaplanMeierFitter
from sklearn.metrics import brier_score_loss
from lifelines.utils import concordance_index
from scipy.stats import chi2
import logging

# Set up logging
logger = logging.getLogger(__name__)

@step
def kfre_eval(train_df: pd.DataFrame,
              spatial_test_df: pd.DataFrame,
              temporal_test_df: pd.DataFrame,
              n_bootstrap: int = 10,
              n_quantiles: int = 10,
              output_path: str = "kfre_evaluation_results.csv") -> Dict:
    """Evaluate KFRE risk predictions on the given datasets.
    
    Args:
        train_df: Training dataset
        spatial_test_df: Spatial test dataset
        temporal_test_df: Temporal test dataset
        n_bootstrap: Number of bootstrap iterations
        n_quantiles: Number of quantiles for risk stratification
        output_path: Path to save the evaluation results CSV
        
    Returns:
        Dictionary containing evaluation results
    """
    # 1. Calculate KFRE risk scores for all datasets
    logger.info("Calculating KFRE risk scores...")
    kfre_calculator = KFRECalculator("src/default_master_df_mapping.yml")
    
    train_df_kfre = kfre_calculator.add_kfre_risk(train_df)
    spatial_test_df_kfre = kfre_calculator.add_kfre_risk(spatial_test_df)
    temporal_test_df_kfre = kfre_calculator.add_kfre_risk(temporal_test_df)
    
    # Define datasets dictionary for easier iteration
    datasets = {
        'train': train_df_kfre,
        'spatial_test': spatial_test_df_kfre,
        'temporal_test': temporal_test_df_kfre
    }
    
    # 2. Initialize results storage
    results = []
    
    # Define risk columns and time points
    risk_cols = ['4v2y', '4v5y', '8v2y', '8v5y']
    time_points = np.array([730, 1825])  # 2 years and 5 years in days
    
    # 3. Bootstrap loop
    for b in range(n_bootstrap):
        logger.info(f"Processing bootstrap {b+1} of {n_bootstrap}...")
        
        # For each dataset
        for dataset_name, dataset in datasets.items():
            # Sample with replacement
            bootstrap_indices = np.random.choice(dataset.index, size=len(dataset), replace=True)
            df_bootstrap = dataset.loc[bootstrap_indices].copy()
            
            # For each time point
            for t in time_points:
                # For each risk column
                for risk_col in risk_cols:
                    try:
                        # Get rows with non-null risk values
                        df_focus = df_bootstrap[df_bootstrap[risk_col].notnull()].copy()
                        
                        # Skip if no data
                        if len(df_focus) == 0:
                            logger.warning(f"No data with non-null {risk_col} values in {dataset_name}")
                            continue
                        
                        # Focus on dialysis events (endpoint = 1)
                        df_focus = df_event_focus(df_focus, 'endpoint', 1)
                        
                        # Calculate null Brier score
                        null_brier = NullModel.calculate_null_brier(
                            df_focus['duration'].values,
                            df_focus['endpoint'].values,
                            time_grid=np.array([t]),
                            event_of_interest=1
                        )
                        
                        # Extract the scalar value from the Series if needed
                        if hasattr(null_brier, 'values') and hasattr(null_brier, 'iloc'):
                            null_brier = null_brier.iloc[0]
                        
                        # Divide into quantiles
                        df_focus['quantile'] = pd.qcut(
                            df_focus[risk_col], 
                            n_quantiles, 
                            labels=False, 
                            duplicates='drop'
                        )
                        
                        # Calculate Brier score for the entire dataset
                        brier_score = brier_score_loss(
                            (df_focus['duration'] <= t) & (df_focus['endpoint'] == 1),
                            df_focus[risk_col]
                        )
                        
                        # Calculate c-index
                        c_index = concordance_index(
                            df_focus['duration'],
                            -df_focus[risk_col],
                            df_focus['endpoint']
                        )
                        
                        # For each quantile
                        for q in df_focus['quantile'].unique():
                            # Get subset for this quantile
                            quantile_df = df_focus[df_focus['quantile'] == q].copy()
                            
                            # Calculate observed risk using Kaplan-Meier
                            kmf = KaplanMeierFitter()
                            kmf.fit(quantile_df['duration'], quantile_df['endpoint'])
                            
                            # Get observed risk at time point
                            if t in kmf.cumulative_density_.index:
                                observed_risk = kmf.cumulative_density_.loc[t].values[0]
                            else:
                                # Find closest time point
                                closest_idx = kmf.cumulative_density_.index.get_indexer([t], method='nearest')[0]
                                closest_time = kmf.cumulative_density_.index[closest_idx]
                                observed_risk = kmf.cumulative_density_.loc[closest_time].values[0]
                            
                            # Calculate predicted risk (mean of KFRE risk in this quantile)
                            predicted_risk = quantile_df[risk_col].mean()
                            
                            # Store results
                            results.append({
                                'dataset': dataset_name,
                                'bootstrap': b,
                                'timepoint': t,
                                'quantile': int(q),
                                'risk': risk_col,
                                'null_brier_score': null_brier,
                                'predicted_risk': predicted_risk,
                                'observed_risk': observed_risk,
                                'brier_score': brier_score,
                                'c_index': c_index,
                                'chi2_stat': np.nan  # Will calculate chi2 after all quantiles are processed
                            })
                        
                        # Calculate chi2 statistic for all quantiles together
                        # Group by quantile
                        grouped = df_focus.groupby('quantile')
                        observed_events = grouped.apply(lambda x: ((x['duration'] <= t) & (x['endpoint'] == 1)).sum())
                        expected_events = grouped[risk_col].sum()
                        n = grouped.size()
                        
                        # Calculate chi2 statistic
                        chi2_stat = np.sum(((observed_events - expected_events) ** 2) / (expected_events * (1 - expected_events / n)))
                        
                        # Update chi2 statistic for all quantiles in this bootstrap/timepoint/risk_col
                        for idx in range(len(results)):
                            if (results[idx]['dataset'] == dataset_name and 
                                results[idx]['bootstrap'] == b and 
                                results[idx]['timepoint'] == t and 
                                results[idx]['risk'] == risk_col):
                                results[idx]['chi2_stat'] = chi2_stat
                    
                    except Exception as e:
                        logger.error(f"Error processing {dataset_name}, bootstrap {b}, time {t}, risk {risk_col}: {e}")
                        continue
    
    # 4. Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # 5. Save to CSV
    # Only create directory if output_path has a directory component
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    logger.info(f"Saved KFRE evaluation results to {output_path}")
    
    # 6. Return summary statistics
    summary = {
        'n_bootstrap': n_bootstrap,
        'n_quantiles': n_quantiles,
        'datasets': list(datasets.keys()),
        'risk_cols': risk_cols,
        'time_points': time_points.tolist(),
        'output_path': output_path,
        'results_shape': results_df.shape
    }
    
    return summary