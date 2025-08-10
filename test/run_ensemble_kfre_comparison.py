"""
Runner script to execute the complete ensemble vs KFRE comparison analysis.
This script runs all necessary steps in sequence.
"""

import os
import sys
import subprocess
from datetime import datetime

def run_command(cmd, description):
    """Run a command and handle output."""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Running: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Success!")
            if result.stdout:
                print("\nOutput:")
                print(result.stdout)
        else:
            print("✗ Failed!")
            if result.stderr:
                print("\nError:")
                print(result.stderr)
            if result.stdout:
                print("\nOutput:")
                print(result.stdout)
            return False
    except Exception as e:
        print(f"✗ Exception occurred: {e}")
        return False
    
    return True

def check_prerequisites():
    """Check if required files and directories exist."""
    print("\nChecking prerequisites...")
    
    checks = {
        "Ensemble CIF files": [
            "results/full_ensemble/temporal_ensemble_cif.h5",
            "results/full_ensemble/spatial_ensemble_cif.h5"
        ],
        "Ground truth labels": [
            "results/final_deploy/temporal_test_labels.csv",
            "results/final_deploy/spatial_test_labels.csv"
        ],
        "YAML mapping": [
            "src/default_master_df_mapping.yml"
        ]
    }
    
    all_good = True
    for category, files in checks.items():
        print(f"\n{category}:")
        for file in files:
            if os.path.exists(file):
                print(f"  ✓ {file}")
            else:
                print(f"  ✗ {file} (missing)")
                all_good = False
    
    return all_good

def main():
    """Main function to run the complete analysis."""
    print("="*80)
    print("ENSEMBLE vs KFRE vs NULL MODEL COMPARISON")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n⚠️  Warning: Some prerequisite files are missing.")
        print("The analysis may fail if these files are required.")
        response = input("\nDo you want to continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            return
    
    # Step 1: Extract test datasets
    print("\n" + "="*80)
    print("STEP 1: Extract Test Datasets")
    print("="*80)
    
    # Check if test datasets already exist (both imputed and preprocessed)
    test_files_exist = (
        os.path.exists("results/final_deploy/temporal_test_df_imputed.pkl") and
        os.path.exists("results/final_deploy/spatial_test_df_imputed.pkl") and
        os.path.exists("results/final_deploy/temporal_test_df_preprocessed.pkl") and
        os.path.exists("results/final_deploy/spatial_test_df_preprocessed.pkl")
    )
    
    if test_files_exist:
        print("Test datasets already exist (both imputed and preprocessed versions).")
        response = input("Do you want to re-extract them? (y/n): ")
        if response.lower() == 'y':
            if not run_command("python extract_test_datasets.py", "Extracting test datasets"):
                print("Failed to extract test datasets. Exiting...")
                return
    else:
        if not run_command("python extract_test_datasets.py", "Extracting test datasets"):
            print("Failed to extract test datasets. Exiting...")
            return
    
    # Step 2: Calculate KFRE predictions
    print("\n" + "="*80)
    print("STEP 2: Calculate KFRE Predictions")
    print("="*80)
    
    # Check if KFRE predictions already exist
    kfre_files_exist = (
        os.path.exists("results/kfre_predictions/temporal_kfre_predictions.csv") and
        os.path.exists("results/kfre_predictions/spatial_kfre_predictions.csv")
    )
    
    if kfre_files_exist:
        print("KFRE predictions already exist.")
        response = input("Do you want to recalculate them? (y/n): ")
        if response.lower() == 'y':
            if not run_command("python calculate_kfre_predictions.py", "Calculating KFRE predictions"):
                print("Failed to calculate KFRE predictions. Exiting...")
                return
    else:
        if not run_command("python calculate_kfre_predictions.py", "Calculating KFRE predictions"):
            print("Failed to calculate KFRE predictions. Exiting...")
            return
    
    # Step 3: Run comparison analysis
    print("\n" + "="*80)
    print("STEP 3: Run Comparison Analysis")
    print("="*80)
    
    if not run_command("python compare_ensemble_kfre_null.py", "Running comparison analysis"):
        print("Failed to run comparison analysis. Exiting...")
        return
    
    # Summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    print("\nResults saved to:")
    print("  - results/ensemble_kfre_comparison/comparison_results.json")
    print("  - results/ensemble_kfre_comparison/*.png (calibration plots)")
    
    # Check if results were created
    results_dir = "results/ensemble_kfre_comparison"
    if os.path.exists(results_dir):
        files = os.listdir(results_dir)
        if files:
            print(f"\nGenerated {len(files)} files:")
            for file in sorted(files):
                print(f"  - {file}")
        else:
            print("\n⚠️  Warning: No files were generated in the results directory.")
    else:
        print("\n⚠️  Warning: Results directory was not created.")

if __name__ == "__main__":
    main()