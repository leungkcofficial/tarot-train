"""
Extract and Save Preprocessing Pipeline

This script extracts all preprocessing parameters from the training data
and saves them in a reusable preprocessor object.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
import argparse

# Import the preprocessor
from src.ckd_preprocessor import CKDPreprocessor

# Import data processing steps from the original pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.merge_data import merge_data
from steps.split_data import split_data


def load_training_data():
    """Load and prepare training data using the existing pipeline steps."""
    print("=== Loading training data ===\n")
    
    # Step 1: Ingest data from multiple sources
    print("Step 1: Ingesting data...")
    cr_df, hb_df, a1c_df, alb_df, po4_df, ca_df, ca_adjusted_df, hco3_df, upcr_df, uacr_df, demo_df, icd10_df, death_df, operation_df = ingest_data()
    
    # Step 2: Clean all dataframes
    print("\nStep 2: Cleaning data...")
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
    print("\nStep 3: Merging data...")
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
    print("\nStep 4: Splitting data...")
    train_df, temporal_test_df, spatial_test_df, raw_df = split_data(
        raw_df=final_df,
        prediction_df=prediction_df
    )
    
    print(f"\nTraining data loaded: {len(train_df)} rows, {train_df['key'].nunique()} unique patients")
    
    return train_df


def extract_and_save_preprocessor(output_path: str = "results/final_deploy/ckd_preprocessor.pkl"):
    """
    Extract preprocessing pipeline from training data and save it.
    
    Args:
        output_path: Path to save the preprocessor
    """
    # Load training data
    train_df = load_training_data()
    
    # Create and fit preprocessor
    print("\n=== Creating and fitting preprocessor ===\n")
    preprocessor = CKDPreprocessor()
    preprocessor.fit(train_df, random_seed=42)
    
    # Save preprocessor
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    preprocessor.save(output_path)
    
    # Print summary
    print("\n=== Preprocessing Pipeline Summary ===")
    info = preprocessor.get_preprocessing_info()
    print(f"\nTotal features: {info['n_features']}")
    print(f"\nImputation:")
    print(f"  - MICE fitted: {info['imputation']['mice_fitted']}")
    print(f"  - Hard truth values: {info['imputation']['n_hard_truth_values']}")
    print(f"  - Medical history values: {info['imputation']['n_medical_history_values']}")
    print(f"\nTransformations:")
    print(f"  - Log transformed: {info['transformations']['n_log_transformed']} columns")
    print(f"  - MinMax scaled: {info['transformations']['n_minmax_scaled']} columns")
    print(f"\nColumn types:")
    for col_type, count in info['column_types'].items():
        print(f"  - {col_type}: {count} columns")
    
    print(f"\n✓ Preprocessor saved to: {output_path}")
    
    # Create a simple test to verify it works
    print("\n=== Testing preprocessor with sample data ===")
    
    # Create sample patient data (with some missing values)
    sample_patient = {
        'key': 'TEST001',
        'date': '2023-06-01',
        'dob': '1960-01-01',
        'gender': 1,
        'creatinine': 1.5,
        'hemoglobin': 12.0,
        'albumin': np.nan,  # Missing value
        'egfr': 45,
        'age': 63,
        'ht': 1,
        'dm': 0
    }
    
    # Load and test
    loaded_preprocessor = CKDPreprocessor.load(output_path)
    
    try:
        # Transform single patient
        transformed = loaded_preprocessor.transform(sample_patient)
        print(f"Sample patient transformed successfully!")
        print(f"Input features: {len(sample_patient)}")
        print(f"Output features: {len(transformed.columns)}")
        print(f"Missing values handled: {transformed.isna().sum().sum()} remaining NaN values")
    except Exception as e:
        print(f"Error during transformation: {e}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Extract preprocessing pipeline from training data')
    parser.add_argument('--output', type=str, 
                       default='results/final_deploy/ckd_preprocessor.pkl',
                       help='Output path for the preprocessor file')
    
    args = parser.parse_args()
    
    print(f"CKD Preprocessing Pipeline Extraction")
    print(f"=====================================")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output path: {args.output}")
    print()
    
    try:
        output_path = extract_and_save_preprocessor(args.output)
        print(f"\n✓ Extraction completed successfully!")
        
        # Print usage instructions
        print("\n" + "="*50)
        print("USAGE INSTRUCTIONS")
        print("="*50)
        print("\nTo use this preprocessor in your code:")
        print("\n```python")
        print("from src.ckd_preprocessor import CKDPreprocessor")
        print("")
        print("# Load the preprocessor")
        print(f"preprocessor = CKDPreprocessor.load('{output_path}')")
        print("")
        print("# Transform new patient data")
        print("patient_data = {")
        print("    'creatinine': 1.5,")
        print("    'hemoglobin': 12.0,")
        print("    'albumin': 3.8,")
        print("    # ... other variables")
        print("}")
        print("")
        print("preprocessed = preprocessor.transform(patient_data)")
        print("```")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())