"""
Generate Predictions with Baseline Hazards Pipeline

This pipeline generates predictions for all models (DeepSurv and DeepHit)
using the preprocessed test data and baseline hazards computed earlier.
It ensures all models produce predictions with consistent shapes.
"""

import os
from zenml.pipelines import pipeline
from typing import Dict, Any

@pipeline(enable_cache=True)
def generate_predictions_pipeline():
    """
    Pipeline for generating predictions with baseline hazards for all models.
    
    This pipeline:
    1. Processes data using the same steps as ensemble_deploy_pipeline
    2. Loads all models from results/final_deploy/models
    3. Loads baseline hazards for DeepSurv models
    4. Generates predictions for both temporal and spatial test sets
    5. Ensures all predictions have shape (2, 5, n_samples) for consistency
    
    Returns:
        Summary of prediction generation results
    """
    # Import utility function for loading YAML files
    from src.util import load_yaml_file
    
    # Import data processing steps (same as ensemble_deploy_pipeline)
    from steps.ingest_data import ingest_data
    from steps.clean_data import clean_data
    from steps.merge_data import merge_data
    from steps.split_data import split_data
    from steps.impute_data import impute_data
    from steps.preprocess_data import preprocess_data
    
    # Import new step for generating predictions
    from steps.generate_all_predictions_with_baseline import generate_all_predictions_with_baseline
    
    # ==================== Data Processing Steps (Cached) ====================
    # These steps are identical to ensemble_deploy_pipeline to ensure consistency
    
    # Step 1: Ingest data from multiple sources
    cr_df, hb_df, a1c_df, alb_df, po4_df, ca_df, ca_adjusted_df, hco3_df, upcr_df, uacr_df, demo_df, icd10_df, death_df, operation_df = ingest_data()
    
    # Step 2: Clean all dataframes
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
    
    # Step 3: Merge cleaned dataframes
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
    
    # Step 4: Split data
    train_df, temporal_test_df, spatial_test_df, raw_df = split_data(
        raw_df=final_df,
        prediction_df=prediction_df
    )

    # Step 5: Impute missing data
    train_df_imputed, temporal_test_df_imputed, spatial_test_df_imputed = impute_data(
        train_df=train_df,
        temporal_test_df=temporal_test_df,
        spatial_test_df=spatial_test_df
    )
    
    # Step 6: Preprocess data
    train_df_preprocessed, temporal_test_df_preprocessed, spatial_test_df_preprocessed = preprocess_data(
        train_df=train_df_imputed,
        temporal_test_df=temporal_test_df_imputed,
        spatial_test_df=spatial_test_df_imputed
    )
    
    # ==================== Prediction Generation ====================
    
    print("\n=== Generating Predictions for All Models with Baseline Hazards ===")
    
    # Step 7: Generate predictions for all models
    prediction_results = generate_all_predictions_with_baseline(
        temporal_test_df_preprocessed=temporal_test_df_preprocessed,
        spatial_test_df_preprocessed=spatial_test_df_preprocessed,
        models_dir="results/final_deploy/models",
        model_config_dir="results/final_deploy/model_config",
        output_dir="results/final_deploy/individual_predictions"
    )
    
    print("\n=== Prediction Generation Pipeline Complete ===")
    
    return prediction_results


if __name__ == "__main__":
    # Run the pipeline
    pipeline_instance = generate_predictions_pipeline()
    pipeline_instance.run()