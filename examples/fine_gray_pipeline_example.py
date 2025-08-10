#!/usr/bin/env python3
"""
Example script demonstrating how to use the r_fine_gray module in a pipeline context
with train_df, spatial_test_df, and temporal_test_df.

This script assumes that these dataframes are already loaded in memory from previous
pipeline steps or can be loaded from a specific source.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Import the r_fine_gray module
from src.r_fine_gray import run_baseline_cif, load_and_predict
from src.util import load_yaml_file

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_datasets() -> Dict[str, pd.DataFrame]:
    """
    Load the train_df, spatial_test_df, and temporal_test_df datasets.
    
    In a real pipeline, these would likely be loaded from a database or previous pipeline step.
    This function is a placeholder for your actual data loading logic.
    
    Returns:
        Dictionary containing the three dataframes
    """
    # Example loading from ZenML artifacts (as seen in test_fine_gray_data.py)
    # Uncomment and modify as needed for your actual data loading
    """
    from zenml.client import Client
    
    logger.info("Loading training data from ZenML artifact...")
    train_artifact = Client().get_artifact_version("your-train-artifact-id")
    train_df = train_artifact.load()
    
    logger.info("Loading spatial test data from ZenML artifact...")
    spatial_artifact = Client().get_artifact_version("your-spatial-test-artifact-id")
    spatial_test_df = spatial_artifact.load()
    
    logger.info("Loading temporal test data from ZenML artifact...")
    temporal_artifact = Client().get_artifact_version("your-temporal-test-artifact-id")
    temporal_test_df = temporal_artifact.load()
    """
    
    # Example loading from CSV files
    # Uncomment and modify as needed for your actual data loading
    """
    train_df = pd.read_csv("path/to/train_df.csv")
    spatial_test_df = pd.read_csv("path/to/spatial_test_df.csv")
    temporal_test_df = pd.read_csv("path/to/temporal_test_df.csv")
    """
    
    # For this example, we'll create dummy dataframes with the expected structure
    # Replace this with your actual data loading code
    
    # Create dummy data with the structure shown in the user's example
    columns = [
        'key', 'date', 'icd10', 'dob', 'gender', 'endpoint', 'endpoint_date',
        'first_sub_60_date', 'hemoglobin', 'calcium', 'duration', 'observation_period',
        'age', 'age_at_obs', 'creatinine', 'a1c', 'albumin', 'phosphate', 'upcr', 'uacr',
        'bicarbonate', 'cci_score_total', 'ht'
    ]
    
    # Create dummy dataframes
    train_df = pd.DataFrame(columns=columns)
    spatial_test_df = pd.DataFrame(columns=columns)
    temporal_test_df = pd.DataFrame(columns=columns)
    
    logger.info(f"Loaded datasets - Train: {train_df.shape}, Spatial: {spatial_test_df.shape}, Temporal: {temporal_test_df.shape}")
    
    return {
        'train_df': train_df,
        'spatial_test_df': spatial_test_df,
        'temporal_test_df': temporal_test_df
    }

def run_fine_gray_analysis(
    train_df: pd.DataFrame,
    spatial_test_df: pd.DataFrame,
    temporal_test_df: pd.DataFrame,
    output_path: str = "./fine_gray_output",
    seed: int = 42,
    n_threads: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run the Fine-Gray analysis on the provided datasets.
    
    Args:
        train_df: Training dataframe
        spatial_test_df: Spatial test dataframe
        temporal_test_df: Temporal test dataframe
        output_path: Path to save outputs
        seed: Random seed for reproducibility
        n_threads: Number of threads to use
        
    Returns:
        Dictionary containing results and predictions
    """
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Load mapping configuration
    mapping_file = "src/default_master_df_mapping.yml"
    mapping = load_yaml_file(mapping_file)
    
    # Get feature columns from mapping
    feature_cols = mapping.get('features', [])
    logger.info(f"Using feature columns: {feature_cols}")
    
    # Step 1: Fit Fine-Gray models on training data
    logger.info("Fitting Fine-Gray models on training data...")
    
    train_results = run_baseline_cif(
        df=train_df,
        feature_cols=feature_cols,
        output_path=output_path,
        seed=seed,
        n_threads=n_threads,
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
        df=spatial_test_df,
        feature_cols=feature_cols,
        time_horizons=time_horizons,
        seed=seed
    )
    
    spatial_results['death_predictions'] = load_and_predict(
        model_path=death_model_path,
        df=spatial_test_df,
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
        df=temporal_test_df,
        feature_cols=feature_cols,
        time_horizons=time_horizons,
        seed=seed
    )
    
    temporal_results['death_predictions'] = load_and_predict(
        model_path=death_model_path,
        df=temporal_test_df,
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
    """Main function to run the Fine-Gray analysis pipeline."""
    logger.info("Starting Fine-Gray analysis pipeline...")
    
    # Step 1: Load datasets
    datasets = load_datasets()
    
    # Step 2: Run Fine-Gray analysis
    results = run_fine_gray_analysis(
        train_df=datasets['train_df'],
        spatial_test_df=datasets['spatial_test_df'],
        temporal_test_df=datasets['temporal_test_df'],
        output_path="./fine_gray_output",
        seed=42
    )
    
    logger.info("Fine-Gray analysis pipeline completed successfully")

if __name__ == "__main__":
    main()