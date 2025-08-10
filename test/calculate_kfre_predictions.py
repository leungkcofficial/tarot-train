"""
Script to calculate KFRE predictions for temporal and spatial test sets.
This script loads test datasets and calculates both 4-variable and 8-variable KFRE predictions.
"""

import pandas as pd
import numpy as np
import h5py
import os
from src.KFRE import KFRECalculator
import pickle
import json
from datetime import datetime

def load_test_datasets():
    """
    Load temporal and spatial test datasets.
    We need the IMPUTED datasets (not preprocessed) for KFRE calculation
    because KFRE requires original clinical values, not scaled values.
    """
    # Try to load IMPUTED datasets from pickle files first
    pickle_paths = {
        'temporal': 'results/final_deploy/temporal_test_df_imputed.pkl',
        'spatial': 'results/final_deploy/spatial_test_df_imputed.pkl'
    }
    
    # Fallback to CSV if pickle not found
    csv_paths = {
        'temporal': 'results/final_deploy/temporal_test_df_imputed.csv',
        'spatial': 'results/final_deploy/spatial_test_df_imputed.csv'
    }
    
    datasets = {}
    
    for dataset_type in ['temporal', 'spatial']:
        # Try pickle first
        if os.path.exists(pickle_paths[dataset_type]):
            print(f"Loading {dataset_type} test data from pickle...")
            with open(pickle_paths[dataset_type], 'rb') as f:
                datasets[dataset_type] = pickle.load(f)
        # Try CSV
        elif os.path.exists(csv_paths[dataset_type]):
            print(f"Loading {dataset_type} test data from CSV...")
            datasets[dataset_type] = pd.read_csv(csv_paths[dataset_type])
        else:
            print(f"Warning: Could not find {dataset_type} test dataset")
            datasets[dataset_type] = None
    
    return datasets['temporal'], datasets['spatial']

def calculate_and_save_kfre_predictions(temporal_df, spatial_df, output_dir='results/kfre_predictions'):
    """
    Calculate KFRE predictions for both test sets and save them.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize KFRE calculator
    print("Initializing KFRE calculator...")
    kfre_calculator = KFRECalculator("src/default_master_df_mapping.yml")
    
    results = {}
    
    # Process each dataset
    for dataset_name, df in [('temporal', temporal_df), ('spatial', spatial_df)]:
        if df is None:
            print(f"Skipping {dataset_name} dataset (not found)")
            continue
            
        print(f"\nProcessing {dataset_name} test set...")
        print(f"Dataset shape: {df.shape}")
        
        # Calculate KFRE predictions
        print(f"Calculating KFRE predictions for {dataset_name} dataset...")
        # Note: You may see warnings about log calculations - this is normal when ACR values are 0
        # The KFRE calculator handles these cases appropriately
        df_with_kfre = kfre_calculator.add_kfre_risk(df)
        
        # Extract KFRE columns
        kfre_columns = ['4v2y', '4v5y', '8v2y', '8v5y']
        kfre_predictions = df_with_kfre[kfre_columns].copy()
        
        # Print summary statistics
        print(f"\n{dataset_name.capitalize()} KFRE predictions summary:")
        print(kfre_predictions.describe())
        print(f"\nMissing values:")
        print(kfre_predictions.isnull().sum())
        
        # Save predictions
        csv_path = os.path.join(output_dir, f'{dataset_name}_kfre_predictions.csv')
        kfre_predictions.to_csv(csv_path, index=False)
        print(f"Saved {dataset_name} KFRE predictions to {csv_path}")
        
        # Also save as HDF5 for consistency with other predictions
        h5_path = os.path.join(output_dir, f'{dataset_name}_kfre_predictions.h5')
        with h5py.File(h5_path, 'w') as f:
            for col in kfre_columns:
                f.create_dataset(col, data=kfre_predictions[col].values)
        print(f"Saved {dataset_name} KFRE predictions to {h5_path}")
        
        # Store results - convert numpy types to Python native types for JSON serialization
        results[dataset_name] = {
            'n_samples': int(len(kfre_predictions)),
            'n_valid_4v2y': int(kfre_predictions['4v2y'].notna().sum()),
            'n_valid_4v5y': int(kfre_predictions['4v5y'].notna().sum()),
            'n_valid_8v2y': int(kfre_predictions['8v2y'].notna().sum()),
            'n_valid_8v5y': int(kfre_predictions['8v5y'].notna().sum()),
            'mean_4v2y': float(kfre_predictions['4v2y'].mean()) if not kfre_predictions['4v2y'].isna().all() else None,
            'mean_4v5y': float(kfre_predictions['4v5y'].mean()) if not kfre_predictions['4v5y'].isna().all() else None,
            'mean_8v2y': float(kfre_predictions['8v2y'].mean()) if not kfre_predictions['8v2y'].isna().all() else None,
            'mean_8v5y': float(kfre_predictions['8v5y'].mean()) if not kfre_predictions['8v5y'].isna().all() else None
        }
    
    # Save summary
    summary_path = os.path.join(output_dir, 'kfre_predictions_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results
        }, f, indent=2)
    print(f"\nSaved summary to {summary_path}")
    
    return results

def main():
    """
    Main function to run KFRE prediction calculation.
    """
    print("="*80)
    print("KFRE Prediction Calculation Script")
    print("="*80)
    
    # Load test datasets
    temporal_df, spatial_df = load_test_datasets()
    
    # If we couldn't load the datasets directly, we need to run the pipeline first
    if temporal_df is None and spatial_df is None:
        print("\nError: Could not find test datasets.")
        print("Please ensure the IMPUTED test datasets are available in one of these locations:")
        print("- results/final_deploy/temporal_test_df_imputed.pkl")
        print("- results/final_deploy/spatial_test_df_imputed.pkl")
        print("- results/final_deploy/temporal_test_df_imputed.csv")
        print("- results/final_deploy/spatial_test_df_imputed.csv")
        print("\nYou may need to run extract_test_datasets.py first.")
        print("\nIMPORTANT: KFRE calculation requires the imputed datasets (with original clinical values),")
        print("           not the preprocessed datasets (which have been scaled).")
        return
    
    # Calculate and save KFRE predictions
    results = calculate_and_save_kfre_predictions(temporal_df, spatial_df)
    
    print("\n" + "="*80)
    print("KFRE prediction calculation completed!")
    print("="*80)

if __name__ == "__main__":
    main()