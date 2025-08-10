"""
Test script for the multi-model deployment pipeline.
This script tests the pipeline with a subset of models first.
"""

import os
import json
import pandas as pd
import numpy as np
import h5py
from datetime import datetime

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the pipeline functions
from pipelines.final_deploy import (
    load_all_model_configurations,
    create_model_from_config,
    generate_model_predictions,
    convert_survival_to_cif,
    extract_time_points,
    stack_deepsurv_predictions,
    ensemble_predictions
)


def test_configuration_loading():
    """Test loading model configurations."""
    print("\n=== Testing Configuration Loading ===")
    
    config_csv_path = "results/final_deploy/model_config/model_config.csv"
    config_dir = "results/final_deploy/model_config"
    
    if not os.path.exists(config_csv_path):
        print(f"ERROR: Configuration CSV not found at {config_csv_path}")
        return None
    
    configs = load_all_model_configurations(config_csv_path, config_dir)
    
    print(f"\nLoaded {len(configs)} configurations")
    
    # Display summary
    deepsurv_count = sum(1 for c in configs if c['algorithm'] == 'DeepSurv')
    deephit_count = sum(1 for c in configs if c['algorithm'] == 'DeepHit')
    
    print(f"DeepSurv models: {deepsurv_count}")
    print(f"DeepHit models: {deephit_count}")
    
    # Check for complete DeepSurv pairs
    deepsurv_groups = {}
    for c in configs:
        if c['algorithm'] == 'DeepSurv':
            key = (c['structure'], c['balancing_method'], c['optimization_target'])
            if key not in deepsurv_groups:
                deepsurv_groups[key] = {'Event 1': False, 'Event 2': False}
            deepsurv_groups[key][c['prediction_endpoint']] = True
    
    complete_groups = sum(1 for g in deepsurv_groups.values() if g['Event 1'] and g['Event 2'])
    print(f"Complete DeepSurv groups: {complete_groups} out of {len(deepsurv_groups)}")
    
    return configs


def test_model_loading(configs, test_models=3):
    """Test loading a subset of models."""
    print(f"\n=== Testing Model Loading (first {test_models} models) ===")
    
    loaded_models = []
    
    for i, config in enumerate(configs[:test_models]):
        print(f"\nLoading model {config['model_no']}: {config['algorithm']} - {config['structure']}")
        
        try:
            model = create_model_from_config(config, device='cpu')
            loaded_models.append((config, model))
            print(f"✓ Successfully loaded model {config['model_no']}")
        except Exception as e:
            print(f"✗ Failed to load model {config['model_no']}: {e}")
    
    return loaded_models


def test_prediction_shapes():
    """Test prediction shape transformations."""
    print("\n=== Testing Prediction Shape Transformations ===")
    
    # Test CIF conversion
    print("\n1. Testing CIF conversion:")
    survival_probs = np.random.rand(1825, 100)  # 1825 time points, 100 samples
    survival_probs = np.sort(survival_probs, axis=0)[::-1]  # Ensure decreasing
    
    cif = convert_survival_to_cif(survival_probs)
    print(f"   Input shape: {survival_probs.shape}")
    print(f"   Output shape: {cif.shape}")
    print(f"   CIF range: [{cif.min():.3f}, {cif.max():.3f}]")
    
    # Test time point extraction
    print("\n2. Testing time point extraction:")
    extracted = extract_time_points(cif)
    print(f"   Input shape: {cif.shape}")
    print(f"   Output shape: {extracted.shape}")
    print(f"   Expected shape: (5, {survival_probs.shape[1]})")
    
    # Test stacking
    print("\n3. Testing prediction stacking:")
    event1_preds = np.random.rand(5, 100)
    event2_preds = np.random.rand(5, 100)
    stacked = np.stack([event1_preds, event2_preds], axis=0)
    print(f"   Event 1 shape: {event1_preds.shape}")
    print(f"   Event 2 shape: {event2_preds.shape}")
    print(f"   Stacked shape: {stacked.shape}")
    
    # Test ensemble
    print("\n4. Testing ensemble:")
    all_preds = np.random.rand(24, 2, 5, 100)  # 24 models, 2 endpoints, 5 times, 100 samples
    ensemble = ensemble_predictions(all_preds, method='average')
    print(f"   Input shape: {all_preds.shape}")
    print(f"   Output shape: {ensemble.shape}")
    print(f"   Expected shape: (2, 5, {all_preds.shape[3]})")


def validate_output_files(timestamp):
    """Validate the output files from a pipeline run."""
    print(f"\n=== Validating Output Files (timestamp: {timestamp}) ===")
    
    individual_dir = "results/final_deploy/individual_predictions"
    ensemble_dir = "results/final_deploy/ensemble_predictions"
    
    # Check directories exist
    if not os.path.exists(individual_dir):
        print(f"✗ Individual predictions directory not found: {individual_dir}")
        return False
    
    if not os.path.exists(ensemble_dir):
        print(f"✗ Ensemble predictions directory not found: {ensemble_dir}")
        return False
    
    # Check ensemble files
    ensemble_files = [
        f"ensemble_temporal_predictions_{timestamp}.h5",
        f"ensemble_spatial_predictions_{timestamp}.h5",
        f"ensemble_temporal_metadata_{timestamp}.csv",
        f"ensemble_spatial_metadata_{timestamp}.csv",
        f"deployment_log_{timestamp}.json"
    ]
    
    all_found = True
    for filename in ensemble_files:
        filepath = os.path.join(ensemble_dir, filename)
        if os.path.exists(filepath):
            print(f"✓ Found: {filename}")
            
            # Validate HDF5 files
            if filename.endswith('.h5'):
                try:
                    with h5py.File(filepath, 'r') as f:
                        if 'predictions' in f:
                            shape = f['predictions'].shape
                            print(f"  - Predictions shape: {shape}")
                            if shape[0] != 2 or shape[1] != 5:
                                print(f"  ✗ WARNING: Unexpected shape! Expected (2, 5, n_samples)")
                        else:
                            print(f"  ✗ WARNING: No 'predictions' dataset found!")
                except Exception as e:
                    print(f"  ✗ ERROR reading HDF5: {e}")
        else:
            print(f"✗ Missing: {filename}")
            all_found = False
    
    # Check deployment log
    log_path = os.path.join(ensemble_dir, f"deployment_log_{timestamp}.json")
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r') as f:
                log_data = json.load(f)
            print(f"\nDeployment Log Summary:")
            print(f"  - Models processed: {log_data.get('n_models_processed', 'N/A')}")
            print(f"  - DeepSurv groups: {log_data.get('n_deepsurv_groups', 'N/A')}")
            print(f"  - DeepHit models: {log_data.get('n_deephit_models', 'N/A')}")
            print(f"  - Ensemble method: {log_data.get('ensemble_method', 'N/A')}")
        except Exception as e:
            print(f"✗ ERROR reading deployment log: {e}")
    
    return all_found


def main():
    """Run all tests."""
    print("=" * 60)
    print("Multi-Model Deployment Pipeline Test Suite")
    print("=" * 60)
    
    # Test 1: Configuration loading
    configs = test_configuration_loading()
    if not configs:
        print("\n✗ Configuration loading failed. Cannot continue tests.")
        return
    
    # Test 2: Model loading (subset)
    loaded_models = test_model_loading(configs, test_models=3)
    
    # Test 3: Prediction shapes
    test_prediction_shapes()
    
    # Test 4: Check for existing outputs
    print("\n=== Checking for Recent Pipeline Outputs ===")
    ensemble_dir = "results/final_deploy/ensemble_predictions"
    if os.path.exists(ensemble_dir):
        files = os.listdir(ensemble_dir)
        log_files = [f for f in files if f.startswith('deployment_log_') and f.endswith('.json')]
        if log_files:
            # Get the most recent
            latest_log = sorted(log_files)[-1]
            timestamp = latest_log.replace('deployment_log_', '').replace('.json', '')
            print(f"Found recent pipeline run: {timestamp}")
            validate_output_files(timestamp)
        else:
            print("No previous pipeline runs found.")
    
    print("\n" + "=" * 60)
    print("Test suite completed!")
    print("=" * 60)
    
    # Recommendations
    print("\nRecommendations:")
    print("1. If all tests pass, run the full pipeline with:")
    print("   python pipelines/final_deploy.py")
    print("2. Monitor GPU memory usage during execution")
    print("3. Check the deployment log after completion")


if __name__ == "__main__":
    main()