#!/usr/bin/env python3
"""
Simple example demonstrating how to use the r_fine_gray module with train_df, 
spatial_test_df, and temporal_test_df.

This script assumes:
1. You have rpy2 and the required R packages installed
2. The dataframes are already loaded in memory or can be loaded from a specific source
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Import the r_fine_gray module
from src.r_fine_gray import run_baseline_cif, load_and_predict
from src.util import load_yaml_file

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def stratified_sample(df, target_size=5000, random_state=42):
    """
    Create a stratified sample of the dataframe based on the endpoint column.
    
    Args:
        df: Input dataframe
        target_size: Target sample size
        random_state: Random seed for reproducibility
        
    Returns:
        Sampled dataframe
    """
    # Set random seed
    np.random.seed(random_state)
    
    # Check if dataframe is empty
    if df.empty or len(df) <= target_size:
        return df
    
    # Get endpoint distribution
    endpoint_col = 'endpoint'  # Adjust if your endpoint column has a different name
    endpoint_counts = df[endpoint_col].value_counts()
    
    # Calculate sampling fraction
    fraction = min(1.0, target_size / len(df))
    
    # Sample with stratification
    sampled_indices = []
    for endpoint_value, count in endpoint_counts.items():
        # Get indices for this endpoint value
        indices = df[df[endpoint_col] == endpoint_value].index.tolist()
        
        # Calculate how many to sample
        n_sample = min(len(indices), int(count * fraction))
        
        # Sample indices
        if n_sample > 0:
            sampled = np.random.choice(indices, n_sample, replace=False)
            sampled_indices.extend(sampled)
    
    # Return sampled dataframe
    logger.info(f"Created stratified sample: {len(sampled_indices)} rows from original {len(df)} rows")
    return df.loc[sampled_indices]

def run_fine_gray_analysis(train_df, spatial_test_df, temporal_test_df, output_path="./fine_gray_output", seed=42, sample_size=5000):
    """
    Run Fine-Gray analysis on the provided dataframes.
    
    Args:
        train_df: Training dataframe
        spatial_test_df: Spatial test dataframe
        temporal_test_df: Temporal test dataframe
        output_path: Path to save outputs
        seed: Random seed for reproducibility
        sample_size: Maximum number of rows to use for model fitting
        
    Returns:
        Dictionary containing results and predictions
    """
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Load mapping configuration
    mapping_file = "/mnt/dump/yard/projects/tarot2/src/default_master_df_mapping.yml"
    mapping = load_yaml_file(mapping_file)
    
    # Get feature columns from mapping
    feature_cols = mapping.get('features', [])
    logger.info(f"Using feature columns: {feature_cols}")
    
    # Sample datasets if they are too large
    logger.info("Checking if datasets need to be sampled...")
    
    if len(train_df) > sample_size:
        logger.info(f"Training dataset is large ({len(train_df)} rows). Creating stratified sample...")
        train_df_sampled = stratified_sample(train_df, target_size=sample_size, random_state=seed)
    else:
        train_df_sampled = train_df
        logger.info(f"Using full training dataset ({len(train_df)} rows)")
    
    if len(spatial_test_df) > sample_size:
        logger.info(f"Spatial test dataset is large ({len(spatial_test_df)} rows). Creating stratified sample...")
        spatial_test_df_sampled = stratified_sample(spatial_test_df, target_size=sample_size, random_state=seed)
    else:
        spatial_test_df_sampled = spatial_test_df
        logger.info(f"Using full spatial test dataset ({len(spatial_test_df)} rows)")
    
    if len(temporal_test_df) > sample_size:
        logger.info(f"Temporal test dataset is large ({len(temporal_test_df)} rows). Creating stratified sample...")
        temporal_test_df_sampled = stratified_sample(temporal_test_df, target_size=sample_size, random_state=seed)
    else:
        temporal_test_df_sampled = temporal_test_df
        logger.info(f"Using full temporal test dataset ({len(temporal_test_df)} rows)")
    
    # Step 1: Fit Fine-Gray models on training data
    logger.info("Fitting Fine-Gray models on training data...")
    
    train_results = run_baseline_cif(
        df=train_df_sampled,
        feature_cols=feature_cols,
        output_path=output_path,
        seed=seed,
        n_threads=None,  # Auto-detect
        silent=False
    )
    
    logger.info("Model fitting completed successfully")
    logger.info(f"Dialysis risks: {train_results['dialysis_risks']}")
    logger.info(f"Death risks: {train_results['death_risks']}")
    
    # Step 2: Load saved models and predict on test datasets
    logger.info("Making predictions on test datasets...")
    
    # Get paths to saved models
    dialysis_model_path = train_results['model_paths']['dialysis_model']
    death_model_path = train_results['model_paths']['death_model']
    
    # Define time horizons
    time_horizons = [365, 730, 1095, 1460, 1825]  # 1-5 years
    
    # Predict on spatial test dataset
    logger.info("Predicting on spatial test dataset...")
    
    spatial_results = {}
    
    spatial_results['dialysis_predictions'] = load_and_predict(
        model_path=dialysis_model_path,
        df=spatial_test_df_sampled,
        feature_cols=feature_cols,
        time_horizons=time_horizons,
        seed=seed
    )
    
    spatial_results['death_predictions'] = load_and_predict(
        model_path=death_model_path,
        df=spatial_test_df_sampled,
        feature_cols=feature_cols,
        time_horizons=time_horizons,
        seed=seed
    )
    
    # Calculate mean risks for spatial test dataset
    spatial_results['dialysis_risks'] = spatial_results['dialysis_predictions'].mean() * 100
    spatial_results['death_risks'] = spatial_results['death_predictions'].mean() * 100
    
    logger.info(f"Spatial test dataset - Dialysis risks: {spatial_results['dialysis_risks'].to_dict()}")
    logger.info(f"Spatial test dataset - Death risks: {spatial_results['death_risks'].to_dict()}")
    
    # Predict on temporal test dataset
    logger.info("Predicting on temporal test dataset...")
    
    temporal_results = {}
    
    temporal_results['dialysis_predictions'] = load_and_predict(
        model_path=dialysis_model_path,
        df=temporal_test_df_sampled,
        feature_cols=feature_cols,
        time_horizons=time_horizons,
        seed=seed
    )
    
    temporal_results['death_predictions'] = load_and_predict(
        model_path=death_model_path,
        df=temporal_test_df_sampled,
        feature_cols=feature_cols,
        time_horizons=time_horizons,
        seed=seed
    )
    
    # Calculate mean risks for temporal test dataset
    temporal_results['dialysis_risks'] = temporal_results['dialysis_predictions'].mean() * 100
    temporal_results['death_risks'] = temporal_results['death_predictions'].mean() * 100
    
    logger.info(f"Temporal test dataset - Dialysis risks: {temporal_results['dialysis_risks'].to_dict()}")
    logger.info(f"Temporal test dataset - Death risks: {temporal_results['death_risks'].to_dict()}")
    
    # Step 3: Save predictions to CSV
    logger.info("Saving predictions to CSV...")
    
    # Save spatial test predictions
    spatial_results['dialysis_predictions'].to_csv(os.path.join(output_path, "spatial_dialysis_predictions.csv"))
    spatial_results['death_predictions'].to_csv(os.path.join(output_path, "spatial_death_predictions.csv"))
    
    # Save temporal test predictions
    temporal_results['dialysis_predictions'].to_csv(os.path.join(output_path, "temporal_dialysis_predictions.csv"))
    temporal_results['death_predictions'].to_csv(os.path.join(output_path, "temporal_death_predictions.csv"))
    
    # Step 4: Create comparison table
    logger.info("Creating risk comparison table...")
    
    # Extract 1-year and 5-year risks
    train_dialysis_1y = train_results['dialysis_risks'][0]['risk_pct']
    train_dialysis_5y = train_results['dialysis_risks'][4]['risk_pct']
    train_death_1y = train_results['death_risks'][0]['risk_pct']
    train_death_5y = train_results['death_risks'][4]['risk_pct']
    
    spatial_dialysis_1y = spatial_results['dialysis_risks']['t365']
    spatial_dialysis_5y = spatial_results['dialysis_risks']['t1825']
    spatial_death_1y = spatial_results['death_risks']['t365']
    spatial_death_5y = spatial_results['death_risks']['t1825']
    
    temporal_dialysis_1y = temporal_results['dialysis_risks']['t365']
    temporal_dialysis_5y = temporal_results['dialysis_risks']['t1825']
    temporal_death_1y = temporal_results['death_risks']['t365']
    temporal_death_5y = temporal_results['death_risks']['t1825']
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        'Dataset': ['Train', 'Spatial Test', 'Temporal Test'],
        'Dialysis 1-year (%)': [train_dialysis_1y, spatial_dialysis_1y, temporal_dialysis_1y],
        'Dialysis 5-year (%)': [train_dialysis_5y, spatial_dialysis_5y, temporal_dialysis_5y],
        'Death 1-year (%)': [train_death_1y, spatial_death_1y, temporal_death_1y],
        'Death 5-year (%)': [train_death_5y, spatial_death_5y, temporal_death_5y]
    })
    
    logger.info(f"Risk comparison:\n{comparison_df}")
    comparison_df.to_csv(os.path.join(output_path, "risk_comparison.csv"), index=False)
    
    # Return all results
    return {
        'train_results': train_results,
        'spatial_results': spatial_results,
        'temporal_results': temporal_results,
        'comparison': comparison_df
    }

def main():
    """
    Main function demonstrating how to use the run_fine_gray_analysis function.
    
    In a real scenario, you would load your actual dataframes here.
    """
    # Example of how to use the function with your dataframes
    
    # Option 1: If your dataframes are already in memory from previous pipeline steps
    # results = run_fine_gray_analysis(train_df, spatial_test_df, temporal_test_df)
    
    # Option 2: If you need to load your dataframes from files
    # train_df = pd.read_csv("path/to/train_df.csv")
    # spatial_test_df = pd.read_csv("path/to/spatial_test_df.csv")
    # temporal_test_df = pd.read_csv("path/to/temporal_test_df.csv")
    # results = run_fine_gray_analysis(train_df, spatial_test_df, temporal_test_df)
    
    # Option 3: If you're using ZenML or another pipeline system
    # from zenml.client import Client
    # train_artifact = Client().get_artifact_version("your-train-artifact-id")
    # train_df = train_artifact.load()
    # spatial_artifact = Client().get_artifact_version("your-spatial-test-artifact-id")
    # spatial_test_df = spatial_artifact.load()
    # temporal_artifact = Client().get_artifact_version("your-temporal-test-artifact-id")
    # temporal_test_df = temporal_artifact.load()
    # results = run_fine_gray_analysis(train_df, spatial_test_df, temporal_test_df)
    
    # For this example, we'll create dummy dataframes
    # In a real scenario, replace this with your actual data loading code
    logger.info("Creating dummy dataframes for demonstration purposes...")
    
    # Create dummy dataframes with the structure shown in the user's example
    columns = [
        'key', 'date', 'icd10', 'dob', 'gender', 'endpoint', 'endpoint_date',
        'first_sub_60_date', 'hemoglobin', 'calcium', 'duration', 'observation_period',
        'age', 'age_at_obs', 'creatinine', 'a1c', 'albumin', 'phosphate', 'upcr', 'uacr',
        'bicarbonate', 'cci_score_total', 'ht'
    ]
    
    # Create dummy dataframes (replace with your actual data loading)
    train_df = pd.DataFrame(columns=columns)
    spatial_test_df = pd.DataFrame(columns=columns)
    temporal_test_df = pd.DataFrame(columns=columns)
    
    logger.info(f"Dataframe shapes - Train: {train_df.shape}, Spatial: {spatial_test_df.shape}, Temporal: {temporal_test_df.shape}")
    
    # Run the analysis
    results = run_fine_gray_analysis(
        train_df,
        spatial_test_df,
        temporal_test_df,
        output_path="./fine_gray_output",
        seed=42,
        sample_size=5000
    )
    
    logger.info("Analysis completed successfully")

if __name__ == "__main__":
    main()