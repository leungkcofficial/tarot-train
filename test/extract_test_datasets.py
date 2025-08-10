"""
Helper script to extract and save test datasets from the pipeline.
This script runs the data processing pipeline up to the point where we have
the preprocessed test datasets, then saves them for use in KFRE calculations.
"""

import os
import sys
import pickle
import pandas as pd
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def extract_and_save_test_datasets():
    """
    Run the pipeline to get test datasets and save them.
    """
    print("Extracting test datasets from pipeline...")
    
    # Import pipeline steps
    from steps.ingest_data import ingest_data
    from steps.clean_data import clean_data
    from steps.merge_data import merge_data
    from steps.split_data import split_data
    from steps.impute_data import impute_data
    from steps.preprocess_data import preprocess_data
    
    # ==================== data processing steps, NOT allowed to change the code below ====================
    # ingest_data now returns multiple DataFrames as separate outputs
    print("Ingesting data...")
    cr_df, hb_df, a1c_df, alb_df, po4_df, ca_df, ca_adjusted_df, hco3_df, upcr_df, uacr_df, demo_df, icd10_df, death_df, operation_df = ingest_data()
    
    # Pass all DataFrames to clean_data, which returns a tuple of cleaned dataframes
    print("Cleaning data...")
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
    print("Merging data...")
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
    print("Splitting data...")
    train_df, temporal_test_df, spatial_test_df, raw_df = split_data(
        raw_df=final_df,
        prediction_df=prediction_df
    )
    
    print("Imputing missing data...")
    train_df_imputed, temporal_test_df_imputed, spatial_test_df_imputed = impute_data(
        train_df=train_df,
        temporal_test_df=temporal_test_df,
        spatial_test_df=spatial_test_df
    )
    
    print("Preprocessing data...")
    train_df_preprocessed, temporal_test_df_preprocessed, spatial_test_df_preprocessed = preprocess_data(
        train_df=train_df_imputed,
        temporal_test_df=temporal_test_df_imputed,
        spatial_test_df=spatial_test_df_imputed
    )
    
    # Create output directory
    output_dir = 'results/final_deploy'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save BOTH imputed and preprocessed test datasets
    print("\nSaving test datasets...")
    
    # Save imputed datasets (for KFRE calculation)
    print("\nSaving imputed datasets (for KFRE calculation)...")
    with open(os.path.join(output_dir, 'temporal_test_df_imputed.pkl'), 'wb') as f:
        pickle.dump(temporal_test_df_imputed, f)
    print(f"  Saved temporal test dataset (imputed): shape {temporal_test_df_imputed.shape}")
    
    with open(os.path.join(output_dir, 'spatial_test_df_imputed.pkl'), 'wb') as f:
        pickle.dump(spatial_test_df_imputed, f)
    print(f"  Saved spatial test dataset (imputed): shape {spatial_test_df_imputed.shape}")
    
    # Also save imputed as CSV for easier inspection
    temporal_test_df_imputed.to_csv(
        os.path.join(output_dir, 'temporal_test_df_imputed.csv'),
        index=False
    )
    spatial_test_df_imputed.to_csv(
        os.path.join(output_dir, 'spatial_test_df_imputed.csv'),
        index=False
    )
    
    # Save preprocessed datasets (for model predictions)
    print("\nSaving preprocessed datasets (for model predictions)...")
    with open(os.path.join(output_dir, 'temporal_test_df_preprocessed.pkl'), 'wb') as f:
        pickle.dump(temporal_test_df_preprocessed, f)
    print(f"  Saved temporal test dataset (preprocessed): shape {temporal_test_df_preprocessed.shape}")
    
    with open(os.path.join(output_dir, 'spatial_test_df_preprocessed.pkl'), 'wb') as f:
        pickle.dump(spatial_test_df_preprocessed, f)
    print(f"  Saved spatial test dataset (preprocessed): shape {spatial_test_df_preprocessed.shape}")
    
    # Also save preprocessed as CSV
    temporal_test_df_preprocessed.to_csv(
        os.path.join(output_dir, 'temporal_test_df_preprocessed.csv'),
        index=False
    )
    spatial_test_df_preprocessed.to_csv(
        os.path.join(output_dir, 'spatial_test_df_preprocessed.csv'),
        index=False
    )
    
    print("\nDatasets saved successfully!")
    print(f"Location: {output_dir}")
    print("\nNote: Use *_imputed.pkl files for KFRE calculation (original clinical values)")
    print("      Use *_preprocessed.pkl files for model predictions (scaled values)")
    
    return temporal_test_df_imputed, spatial_test_df_imputed, temporal_test_df_preprocessed, spatial_test_df_preprocessed

def main():
    """
    Main function to extract test datasets.
    """
    print("="*80)
    print("Test Dataset Extraction Script")
    print("="*80)
    
    try:
        temporal_imputed, spatial_imputed, temporal_preprocessed, spatial_preprocessed = extract_and_save_test_datasets()
        
        print("\n" + "="*80)
        print("Extraction completed successfully!")
        print("="*80)
        
        # Print some basic statistics
        print("\nDataset Statistics:")
        print(f"Temporal test (imputed): {temporal_imputed.shape[0]} samples, {temporal_imputed.shape[1]} features")
        print(f"Spatial test (imputed): {spatial_imputed.shape[0]} samples, {spatial_imputed.shape[1]} features")
        print(f"Temporal test (preprocessed): {temporal_preprocessed.shape[0]} samples, {temporal_preprocessed.shape[1]} features")
        print(f"Spatial test (preprocessed): {spatial_preprocessed.shape[0]} samples, {spatial_preprocessed.shape[1]} features")
        
        # Check for required columns in imputed data
        required_cols = ['duration', 'endpoint']
        print("\nChecking required columns in imputed data:")
        for col in required_cols:
            if col in temporal_imputed.columns:
                print(f"  ✓ '{col}' column found")
            else:
                print(f"  ✗ '{col}' column missing")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Failed to extract test datasets")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()