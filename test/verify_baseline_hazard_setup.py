"""
Verification script to check if everything is set up correctly for the baseline hazard computation.
"""

import os
import glob
import json
from pathlib import Path


def check_file_exists(filepath, description):
    """Check if a file exists and print status."""
    exists = os.path.exists(filepath)
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {filepath}")
    return exists


def check_models():
    """Check if all DeepSurv models exist."""
    print("\n=== Checking DeepSurv Models ===")
    models_dir = "results/final_deploy/models"
    
    all_exist = True
    for model_no in range(1, 25):
        pattern = f"Ensemble_model{model_no}_DeepSurv_*.pt"
        files = glob.glob(os.path.join(models_dir, pattern))
        
        if files:
            print(f"✓ Model {model_no}: {os.path.basename(files[0])}")
        else:
            print(f"✗ Model {model_no}: NOT FOUND")
            all_exist = False
    
    return all_exist


def check_configs():
    """Check if all model configuration files exist."""
    print("\n=== Checking Model Configurations ===")
    config_dir = "results/final_deploy/model_config"
    
    all_exist = True
    for model_no in range(1, 25):
        pattern = f"model{model_no}_details_*.json"
        files = glob.glob(os.path.join(config_dir, pattern))
        
        if files:
            # Load and check if it's a DeepSurv model
            with open(files[0], 'r') as f:
                config = json.load(f)
            
            model_type = config.get('model_type', '').lower()
            network_type = config.get('network_type', 'ann').upper()
            
            if model_type == 'deepsurv':
                if network_type == 'LSTM':
                    seq_len = config.get('sequence_length', 'NOT FOUND')
                    print(f"✓ Model {model_no} config: {os.path.basename(files[0])} (LSTM, seq_len={seq_len})")
                else:
                    print(f"✓ Model {model_no} config: {os.path.basename(files[0])} (ANN)")
            else:
                print(f"⚠ Model {model_no} config: {os.path.basename(files[0])} (type={model_type}, expected deepsurv)")
        else:
            print(f"✗ Model {model_no} config: NOT FOUND")
            all_exist = False
    
    return all_exist


def check_pipeline_files():
    """Check if all pipeline files exist."""
    print("\n=== Checking Pipeline Files ===")
    
    files_to_check = [
        ("pipelines/compute_baseline_hazard.py", "Main pipeline"),
        ("steps/compute_all_baseline_hazards.py", "Baseline hazard computation step"),
        ("run_baseline_hazard_computation.py", "Pipeline runner script"),
        ("test_baseline_hazard_single_model.py", "Test script"),
        ("src/nn_architectures.py", "Neural network architectures"),
        ("src/sequence_utils.py", "Sequence utilities for LSTM")
    ]
    
    all_exist = True
    for filepath, description in files_to_check:
        exists = check_file_exists(filepath, description)
        all_exist = all_exist and exists
    
    return all_exist


def check_existing_baseline_hazards():
    """Check if any baseline hazards already exist."""
    print("\n=== Checking Existing Baseline Hazards ===")
    
    pattern = "results/final_deploy/models/baseline_hazards_*.pkl"
    existing_files = glob.glob(pattern)
    
    if existing_files:
        print(f"Found {len(existing_files)} existing baseline hazard files:")
        for f in existing_files[:5]:  # Show first 5
            print(f"  - {os.path.basename(f)}")
        if len(existing_files) > 5:
            print(f"  ... and {len(existing_files) - 5} more")
    else:
        print("No existing baseline hazard files found (this is expected)")
    
    return len(existing_files)


def main():
    """Run all verification checks."""
    print("BASELINE HAZARD COMPUTATION SETUP VERIFICATION")
    print("=" * 60)
    
    # Check all components
    models_ok = check_models()
    configs_ok = check_configs()
    pipeline_ok = check_pipeline_files()
    existing_count = check_existing_baseline_hazards()
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    all_ok = models_ok and configs_ok and pipeline_ok
    
    if all_ok:
        print("✓ All checks passed!")
        print("\nYou can now run the pipeline with:")
        print("  python run_baseline_hazard_computation.py")
        
        if existing_count > 0:
            print(f"\nNote: Found {existing_count} existing baseline hazard files.")
            print("The pipeline will create new ones with current timestamps.")
    else:
        print("✗ Some checks failed!")
        print("\nPlease fix the issues above before running the pipeline.")
        
        if not models_ok:
            print("\n- Missing model files in results/final_deploy/models/")
        if not configs_ok:
            print("\n- Missing configuration files in results/final_deploy/model_config/")
        if not pipeline_ok:
            print("\n- Missing pipeline implementation files")


if __name__ == "__main__":
    main()