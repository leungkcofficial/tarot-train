"""
Ensemble and deploy Pipeline for CKD Risk Prediction

This module defines the ZenML pipeline for training the CKD risk prediction model.
"""

import os
import numpy as np
import pandas as pd
from zenml.pipelines import pipeline
from typing import Dict, Any, Optional
import glob

@pipeline(enable_cache=True)
def ensemble_pipeline():
    """
    Pipeline for training the CKD risk prediction model.
    
    This pipeline connects the following steps:
    1. ingest_data: Loads lab data and demographic data for multiple lab tests
       - Creatinine, hemoglobin, hemoglobin A1c, albumin, phosphate
       - Calcium and adjusted calcium
       - Urine protein creatinine ratio and urine albumin creatinine ratio
    2. clean_data: Cleans and preprocesses the data
    3. merge_data: Merges all data sources into a master dataframe
    4. split_data: Splits data into training and test sets
    5. impute_data: Imputes missing values
    6. preprocess_data: Preprocesses data for model training
    7. multiple model creation: Creates multiple models with different configurations
    8. train_model: Trains the model with hyperparameter optimization
    9. deploy_model: Deploys the trained model, saves models to predefined paths
    10. make_predictions: Makes predictions using the deployed model
    11. ensemble_predictions: iterates through all models, loads their predictions, and creates an ensemble prediction with different combinations
    12. eval_model: Evaluates the performance of ensembled predictions
    
    Returns:
        Model details and optimization metrics
    """
    # Import utility function for loading YAML files
    from src.util import load_yaml_file
    
    # Load hyperparameter configuration from YAML file
    hyperparameter_config = load_yaml_file("src/hyperparameter_config.yml")
    
    # Get model type from configuration
    model_type = hyperparameter_config.get("model_type", "deepsurv")
    print(f"Using model type from configuration: {model_type}")
    
    # Validate model type
    if model_type.lower() not in ["deepsurv", "deephit"]:
        raise ValueError(f"Unsupported model type in configuration: {model_type}. Supported types are 'deepsurv' and 'deephit'.")
    
    # Import steps here to avoid circular imports
    from steps.ingest_data import ingest_data
    from steps.clean_data import clean_data
    from steps.merge_data import merge_data
    from steps.split_data import split_data
    from steps.impute_data import impute_data
    from steps.preprocess_data import preprocess_data
    from steps.model_train import train_model
    from steps.model_deploy import deploy_model
    from steps.model_eval import eval_model
    
    # ==================== data processing steps, NOT allowed to change the code below ====================
    # ingest_data now returns multiple DataFrames as separate outputs
    cr_df, hb_df, a1c_df, alb_df, po4_df, ca_df, ca_adjusted_df, hco3_df, upcr_df, uacr_df, demo_df, icd10_df, death_df, operation_df = ingest_data()
    
    # Pass all DataFrames to clean_data, which returns a tuple of cleaned dataframes
    patient_df, icd10_df_clean, cr_df_clean, hb_df_clean, a1c_df_clean, alb_df_clean, po4_df_clean, ca_df_clean, ca_adjusted_df_clean, hco3_df_clean, upcr_df_clean, uacr_df_clean, operation_df_clean, death_df_clean, cci_df, cci_score_df, hypertension_df, egfr_df = clean_data(
        cr_df=cr_df,
        hb_df=hb_df,
        a1c_df=a1c_df,
        alb_df=alb_df,
        po4_df=po4_df,
        ca_df=ca_df,
        ca_adjusted_df=ca_adjusted_df,
        hco3_df=hco3_df,
        upcr_df=upcr_df,
        uacr_df=uacr_df,
        icd10_df=icd10_df,
        operation_df=operation_df,
        death_df=death_df,
        demo_df=demo_df
    )
    
    # Merge the cleaned dataframes into a master dataframe
    final_df, prediction_df = merge_data(
        patient_df=patient_df,
        icd10_df=icd10_df_clean,
        cr_df=cr_df_clean,
        hb_df=hb_df_clean,
        a1c_df=a1c_df_clean,
        alb_df=alb_df_clean,
        po4_df=po4_df_clean,
        ca_df=ca_df_clean,
        ca_adjusted_df=ca_adjusted_df_clean,
        hco3_df=hco3_df_clean,
        upcr_df=upcr_df_clean,
        uacr_df=uacr_df_clean,
        operation_df=operation_df_clean,
        death_df=death_df_clean,
        cci_df=cci_df,
        cci_score_df=cci_score_df,
        hypertension_df=hypertension_df,
        egfr_df=egfr_df
    )
    
    # Split the merged data into training, temporal test, and spatial test sets
    train_df, temporal_test_df, spatial_test_df, raw_df = split_data(
        raw_df=final_df,
        prediction_df=prediction_df
    )

    # Impute missing data via MICE
    train_df_imputed, temporal_test_df_imputed, spatial_test_df_imputed = impute_data(
        train_df=train_df,
        temporal_test_df=temporal_test_df,
        spatial_test_df=spatial_test_df
    )
    
    # Preprocess data for model training
    train_df_preprocessed, temporal_test_df_preprocessed, spatial_test_df_preprocessed = preprocess_data(
        train_df=train_df_imputed,
        temporal_test_df=temporal_test_df_imputed,
        spatial_test_df=spatial_test_df_imputed
    )
    
    # Define the path to the master dataframe mapping
    master_df_mapping_path = "src/default_master_df_mapping.yml"
    print(f"Using master dataframe mapping from: {master_df_mapping_path}")
    # ==================== data processing steps, NOT allowed to change the code above ====================
    
    # Now we got 1 training set, 1 temporal test set, and 1 spatial test set
    # use the training set to train the model only, DO NOT use temporal or spatial test set for training
    # the model structure, config and optimization detail
    
    # Import our custom steps
    from steps.load_model_configurations import load_model_configurations
    from steps.process_models_sequentially import process_models_sequentially
    from steps.stack_predictions import stack_predictions
    # from steps.ensemble_predictions import ensemble_predictions
    # from steps.ensemble_evaluator import ensemble_evaluator
    
    # Step 1: Load all model configurations and save to JSON
    print("\n=== Step 1: Loading Model Configurations ===")
    model_configs, config_json_path = load_model_configurations(
        config_csv_path="results/final_deploy/model_config/model_config.csv",
        config_dir="results/final_deploy/model_config",
        output_dir="results/final_deploy/temp"
    )
    
    # Step 2: Process models sequentially
    print("\n=== Step 2: Processing Models Sequentially ===")
    all_deployment_details, all_temporal_predictions, all_spatial_predictions = process_models_sequentially(
        config_json_path=config_json_path,
        train_df_preprocessed=train_df_preprocessed,
        temporal_test_df_preprocessed=temporal_test_df_preprocessed,
        spatial_test_df_preprocessed=spatial_test_df_preprocessed,
        master_df_mapping_path=master_df_mapping_path
    )
    
    # Step 3: Stack predictions (group DeepSurv by event type)
    print("\n=== Step 3: Stacking Predictions ===")
    temporal_stacked, spatial_stacked = stack_predictions(
        deployment_details=all_deployment_details,
        temporal_predictions_paths=all_temporal_predictions,
        spatial_predictions_paths=all_spatial_predictions
    )
    
    # # Step 4: Ensemble predictions (simple averaging)
    # print("\n=== Step 4: Creating Ensemble Predictions ===")
    # temporal_ensemble_path, spatial_ensemble_path, ensemble_metadata = ensemble_predictions(
    #     temporal_stacked=temporal_stacked,
    #     spatial_stacked=spatial_stacked,
    #     ensemble_method="average",
    #     output_dir="results/final_deploy/ensemble_predictions"
    # )
    
    # # Step 5: Evaluate ensemble performance
    # print("\n=== Step 5: Evaluating Ensemble Performance ===")
    # evaluation_results = ensemble_evaluator(
    #     temporal_ensemble_path=temporal_ensemble_path,
    #     spatial_ensemble_path=spatial_ensemble_path,
    #     ensemble_metadata=ensemble_metadata,
    #     temporal_test_df=temporal_test_df,
    #     spatial_test_df=spatial_test_df,
    #     master_df_mapping_path=master_df_mapping_path,
    #     output_dir="results/final_deploy/ensemble_eval"
    # )
    
    print("\n=== Ensemble Pipeline Complete (Steps 1-2) ===")
    
    return {
        "model_configs": model_configs,
        "config_json_path": config_json_path,
        "deployment_details": all_deployment_details,
        "temporal_predictions": all_temporal_predictions,
        "spatial_predictions": all_spatial_predictions
    }