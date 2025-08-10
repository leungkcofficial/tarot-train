"""
Extract test labels (duration and event) for temporal and spatial datasets.
"""

import os
import pickle
import pandas as pd
import numpy as np
import yaml

def load_master_df_mapping():
    """Load the master dataframe mapping configuration."""
    with open('src/default_master_df_mapping.yml', 'r') as f:
        return yaml.safe_load(f)

def extract_labels_from_df(df, mapping):
    """Extract duration and event labels from dataframe."""
    duration_col = mapping['duration']  # 'duration'
    event_col = mapping['event']  # 'endpoint'
    
    # Create labels dataframe
    labels = pd.DataFrame({
        'time': df[duration_col],
        'event': df[event_col]
    })
    
    # Map event values according to the endpoint mapping
    # 0: censored, 1: dialysis, 2: mortality
    # This is already the correct mapping in the data
    
    return labels

def main():
    """Extract and save test labels."""
    
    # Load mapping configuration
    mapping = load_master_df_mapping()
    print("Loaded dataframe mapping configuration")
    print(f"Duration column: {mapping['duration']}")
    print(f"Event column: {mapping['event']}")
    print(f"Event mapping: {mapping['endpoint']}")
    
    # We need to run the data processing pipeline to get the test sets
    # Import the necessary steps
    from steps.ingest_data import ingest_data
    from steps.clean_data import clean_data
    from steps.merge_data import merge_data
    from steps.split_data import split_data
    from steps.impute_data import impute_data
    from steps.preprocess_data import preprocess_data
    
    print("\nRunning data processing pipeline...")
    
    # Step 1: Ingest data
    print("Step 1: Ingesting data...")
    cr_df, hb_df, a1c_df, alb_df, po4_df, ca_df, ca_adjusted_df, hco3_df, \
    upcr_df, uacr_df, icd10_df, death_df, operation_df, demo_df, \
    patient_df, cci_df, cci_score_df, hypertension_df, egfr_df = ingest_data()
    
    # Step 2: Clean data
    print("Step 2: Cleaning data...")
    cr_df_clean, hb_df_clean, a1c_df_clean, alb_df_clean, po4_df_clean, \
    ca_df_clean, ca_adjusted_df_clean, hco3_df_clean, upcr_df_clean, \
    uacr_df_clean, icd10_df_clean, operation_df_clean, death_df_clean = clean_data(
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
    print("Step 3: Merging data...")
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
    print("Step 4: Splitting data...")
    train_df, temporal_test_df, spatial_test_df, raw_df = split_data(
        raw_df=final_df,
        prediction_df=prediction_df
    )
    
    # Extract labels from test sets
    print("\nExtracting test labels...")
    
    # Extract temporal test labels
    y_temporal_test = extract_labels_from_df(temporal_test_df, mapping)
    print(f"\nTemporal test samples: {len(y_temporal_test)}")
    print(f"Event distribution:")
    print(y_temporal_test['event'].value_counts().sort_index())
    
    # Extract spatial test labels
    y_spatial_test = extract_labels_from_df(spatial_test_df, mapping)
    print(f"\nSpatial test samples: {len(y_spatial_test)}")
    print(f"Event distribution:")
    print(y_spatial_test['event'].value_counts().sort_index())
    
    # Save labels
    output_dir = "results/final_deploy"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save temporal test labels
    temporal_path = os.path.join(output_dir, "temporal_test_labels.pkl")
    with open(temporal_path, 'wb') as f:
        pickle.dump(y_temporal_test, f)
    print(f"\nSaved temporal test labels to: {temporal_path}")
    
    # Save spatial test labels
    spatial_path = os.path.join(output_dir, "spatial_test_labels.pkl")
    with open(spatial_path, 'wb') as f:
        pickle.dump(y_spatial_test, f)
    print(f"Saved spatial test labels to: {spatial_path}")
    
    # Print sample of labels
    print("\nSample of temporal test labels:")
    print(y_temporal_test.head(10))
    
    print("\nSample of spatial test labels:")
    print(y_spatial_test.head(10))
    
    # Print summary statistics
    print("\nTemporal test summary:")
    print(f"  Min duration: {y_temporal_test['time'].min()}")
    print(f"  Max duration: {y_temporal_test['time'].max()}")
    print(f"  Mean duration: {y_temporal_test['time'].mean():.2f}")
    
    print("\nSpatial test summary:")
    print(f"  Min duration: {y_spatial_test['time'].min()}")
    print(f"  Max duration: {y_spatial_test['time'].max()}")
    print(f"  Mean duration: {y_spatial_test['time'].mean():.2f}")


if __name__ == "__main__":
    main()