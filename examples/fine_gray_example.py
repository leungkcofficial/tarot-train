#!/usr/bin/env python3
"""
Example script demonstrating how to use the r_fine_gray module to:
1. Fit a Fine-Gray model on training data
2. Save the model
3. Load the model and make predictions on spatial and temporal test datasets
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

def main():
    """Main function to demonstrate Fine-Gray model fitting and prediction."""
    # Define paths
    output_path = Path("./fine_gray_output")
    output_path.mkdir(exist_ok=True)
    
    # Load mapping configuration
    mapping_file = "src/default_master_df_mapping.yml"
    mapping = load_yaml_file(mapping_file)
    
    # Get feature columns from mapping
    feature_cols = mapping.get('features', [])
    logger.info(f"Using feature columns: {feature_cols}")
    
    # Step 1: Load datasets
    # In a real pipeline, these would likely be loaded from a database or previous pipeline step
    logger.info("Loading datasets...")
    
    # Load your datasets here
    # For example:
    # train_df = pd.read_csv("path/to/train_df.csv")
    # spatial_test_df = pd.read_csv("path/to/spatial_test_df.csv")
    # temporal_test_df = pd.read_csv("path/to/temporal_test_df.csv")
    
    # Alternatively, if the dataframes are already in memory from previous pipeline steps:
    # train_df = previous_pipeline_step_train()
    # spatial_test_df = previous_pipeline_step_spatial()
    # temporal_test_df = previous_pipeline_step_temporal()
    
    # For this example, we'll assume the dataframes are already in memory
    # Replace these lines with your actual data loading code
    train_df = pd.DataFrame()  # Replace with your actual train_df
    spatial_test_df = pd.DataFrame()  # Replace with your actual spatial_test_df
    temporal_test_df = pd.DataFrame()  # Replace with your actual temporal_test_df
    
    logger.info(f"Loaded datasets - Train: {train_df.shape}, Spatial: {spatial_test_df.shape}, Temporal: {temporal_test_df.shape}")
    
    # Step 2: Fit Fine-Gray models on training data
    logger.info("Fitting Fine-Gray models on training data...")
    
    results = run_baseline_cif(
        df=train_df,
        feature_cols=feature_cols,
        output_path=str(output_path),
        seed=42,
        n_threads=None,  # Auto-detect
        silent=False
    )
    
    logger.info("Model fitting completed successfully")
    logger.info(f"Dialysis risks: {results['dialysis_risks']}")
    logger.info(f"Death risks: {results['death_risks']}")
    
    # Step 3: Load saved models and predict on test datasets
    logger.info("Making predictions on test datasets...")
    
    # Get paths to saved models
    dialysis_model_path = results['model_paths']['dialysis_model']
    death_model_path = results['model_paths']['death_model']
    
    # Predict on spatial test dataset
    logger.info("Predicting on spatial test dataset...")
    
    spatial_dialysis_predictions = load_and_predict(
        model_path=dialysis_model_path,
        df=spatial_test_df,
        feature_cols=feature_cols,
        time_horizons=[365, 730, 1095, 1460, 1825],  # 1-5 years
        seed=42
    )
    
    spatial_death_predictions = load_and_predict(
        model_path=death_model_path,
        df=spatial_test_df,
        feature_cols=feature_cols,
        time_horizons=[365, 730, 1095, 1460, 1825],  # 1-5 years
        seed=42
    )
    
    # Calculate mean risks for spatial test dataset
    spatial_dialysis_risks = spatial_dialysis_predictions.mean() * 100
    spatial_death_risks = spatial_death_predictions.mean() * 100
    
    logger.info(f"Spatial test dataset - Dialysis risks: {spatial_dialysis_risks.to_dict()}")
    logger.info(f"Spatial test dataset - Death risks: {spatial_death_risks.to_dict()}")
    
    # Predict on temporal test dataset
    logger.info("Predicting on temporal test dataset...")
    
    temporal_dialysis_predictions = load_and_predict(
        model_path=dialysis_model_path,
        df=temporal_test_df,
        feature_cols=feature_cols,
        time_horizons=[365, 730, 1095, 1460, 1825],  # 1-5 years
        seed=42
    )
    
    temporal_death_predictions = load_and_predict(
        model_path=death_model_path,
        df=temporal_test_df,
        feature_cols=feature_cols,
        time_horizons=[365, 730, 1095, 1460, 1825],  # 1-5 years
        seed=42
    )
    
    # Calculate mean risks for temporal test dataset
    temporal_dialysis_risks = temporal_dialysis_predictions.mean() * 100
    temporal_death_risks = temporal_death_predictions.mean() * 100
    
    logger.info(f"Temporal test dataset - Dialysis risks: {temporal_dialysis_risks.to_dict()}")
    logger.info(f"Temporal test dataset - Death risks: {temporal_death_risks.to_dict()}")
    
    # Step 4: Save predictions to CSV (optional)
    logger.info("Saving predictions to CSV...")
    
    # Save spatial test predictions
    spatial_dialysis_predictions.to_csv(output_path / "spatial_dialysis_predictions.csv")
    spatial_death_predictions.to_csv(output_path / "spatial_death_predictions.csv")
    
    # Save temporal test predictions
    temporal_dialysis_predictions.to_csv(output_path / "temporal_dialysis_predictions.csv")
    temporal_death_predictions.to_csv(output_path / "temporal_death_predictions.csv")
    
    logger.info("Predictions saved successfully")
    
    # Step 5: Compare risks across datasets (optional)
    logger.info("Comparing risks across datasets...")
    
    # Create a DataFrame to compare risks
    comparison_df = pd.DataFrame({
        'Dataset': ['Train', 'Spatial Test', 'Temporal Test'],
        'Dialysis 1-year (%)': [
            results['dialysis_risks'][0]['risk_pct'],
            spatial_dialysis_risks['t365'],
            temporal_dialysis_risks['t365']
        ],
        'Dialysis 5-year (%)': [
            results['dialysis_risks'][4]['risk_pct'],
            spatial_dialysis_risks['t1825'],
            temporal_dialysis_risks['t1825']
        ],
        'Death 1-year (%)': [
            results['death_risks'][0]['risk_pct'],
            spatial_death_risks['t365'],
            temporal_death_risks['t365']
        ],
        'Death 5-year (%)': [
            results['death_risks'][4]['risk_pct'],
            spatial_death_risks['t1825'],
            temporal_death_risks['t1825']
        ]
    })
    
    logger.info(f"Risk comparison:\n{comparison_df}")
    comparison_df.to_csv(output_path / "risk_comparison.csv", index=False)
    
    logger.info("Fine-Gray analysis completed successfully")

if __name__ == "__main__":
    main()