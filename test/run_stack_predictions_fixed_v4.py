"""
Fixed script to run stack predictions and ensemble from cached artifacts.
This version handles the deployment details structure properly.
"""

import os
import sys
from zenml.client import Client
from steps.stack_predictions import stack_predictions
from steps.ensemble_predictions_fixed import ensemble_predictions_fixed

# Initialize ZenML client
client = Client()

# Get the cached artifacts from the expensive process_models_sequentially step
print("Loading cached artifacts from process_models_sequentially step...")

# These are the artifact IDs from the 3-day run
deployment_details_artifact_id = "36efa22f-602c-446b-9a3f-86ffb73ee183"

# Load the deployment details artifact
deployment_details_artifact = client.get_artifact_version(deployment_details_artifact_id)
deployment_details_raw = deployment_details_artifact.load()

print(f"\nLoaded deployment details")
print(f"Type: {type(deployment_details_raw)}")

# Convert to list if it's a dict
if isinstance(deployment_details_raw, dict):
    print(f"Deployment details is a dict with keys: {list(deployment_details_raw.keys())}")
    # It might be a dict with numeric keys
    deployment_details = []
    for key in sorted(deployment_details_raw.keys()):
        deployment_details.append(deployment_details_raw[key])
    print(f"Converted to list with {len(deployment_details)} items")
elif isinstance(deployment_details_raw, list):
    deployment_details = deployment_details_raw
    print(f"Deployment details is already a list with {len(deployment_details)} items")
else:
    print(f"Unexpected type for deployment details: {type(deployment_details_raw)}")
    # Try to iterate anyway
    deployment_details = list(deployment_details_raw)

# Debug: Check the structure of deployment details
if deployment_details:
    print("\nSample deployment detail structure:")
    sample = deployment_details[0]
    print(f"Type of first item: {type(sample)}")
    
    if isinstance(sample, dict):
        print(f"Keys: {list(sample.keys())}")
        
        # Check for prediction paths
        if 'temporal_predictions_path' in sample:
            print(f"temporal_predictions_path: {sample['temporal_predictions_path']}")
        if 'spatial_predictions_path' in sample:
            print(f"spatial_predictions_path: {sample['spatial_predictions_path']}")

# Extract the correct prediction paths from deployment details
temporal_paths = []
spatial_paths = []

for i, detail in enumerate(deployment_details):
    if isinstance(detail, dict):
        # The paths might be stored directly in the detail dict
        temp_path = detail.get('temporal_predictions_path', '')
        spat_path = detail.get('spatial_predictions_path', '')
        
        if temp_path:
            temporal_paths.append(temp_path)
        if spat_path:
            spatial_paths.append(spat_path)
        
        # Debug first few entries
        if i < 3:
            print(f"\nDetail {i}:")
            if 'model_config' in detail:
                model_config = detail['model_config']
                print(f"  Model: {model_config.get('algorithm', 'Unknown')} - {model_config.get('structure', 'Unknown')}")
            print(f"  Temporal path: {temp_path}")
            print(f"  Spatial path: {spat_path}")

print(f"\nExtracted prediction paths:")
print(f"Temporal paths: {len(temporal_paths)}")
print(f"Spatial paths: {len(spatial_paths)}")

# If paths are still empty, check if they're nested differently
if not temporal_paths and deployment_details:
    print("\nNo paths found in deployment details. Checking for alternative structure...")
    # Let's check the full structure of the first detail
    import json
    print("\nFull structure of first deployment detail:")
    try:
        print(json.dumps(deployment_details[0], indent=2, default=str))
    except:
        print(f"Could not serialize to JSON. Keys: {deployment_details[0].keys() if hasattr(deployment_details[0], 'keys') else 'N/A'}")

# Show sample counts for each model if we have paths
if temporal_paths:
    print("\nChecking sample counts in prediction files...")
    import h5py
    import numpy as np
    
    for i in range(min(5, len(temporal_paths))):
        temp_path = temporal_paths[i]
        detail = deployment_details[i]
        
        if os.path.exists(temp_path):
            with h5py.File(temp_path, 'r') as f:
                if 'predictions' in f:
                    shape = f['predictions'].shape
                elif 'cif' in f:
                    shape = f['cif'].shape
                else:
                    shape = "Unknown"
                
                model_config = detail.get('model_config', {})
                model_type = model_config.get('algorithm', 'Unknown')
                structure = model_config.get('structure', 'Unknown')
                
                print(f"Model {i}: {model_type} - {structure} - Shape: {shape}")
        else:
            print(f"Model {i}: Path does not exist: {temp_path}")

    # Run the stack predictions step
    print("\n=== Running Stack Predictions Step ===")
    temporal_stacked, spatial_stacked = stack_predictions(
        deployment_details=deployment_details,
        temporal_predictions_paths=temporal_paths,
        spatial_predictions_paths=spatial_paths
    )
    
    print(f"\nStacking complete!")
    print(f"Temporal stacked models: {len(temporal_stacked)}")
    print(f"Spatial stacked models: {len(spatial_stacked)}")
    
    # Show the shapes of stacked predictions
    print("\nStacked prediction shapes:")
    for i, (model_name, predictions) in enumerate(temporal_stacked.items()):
        if i < 5:
            print(f"  {model_name}: {predictions.shape}")
    
    # Continue with the FIXED ensemble predictions
    print("\n=== Running Fixed Ensemble Predictions Step ===")
    temporal_ensemble_path, spatial_ensemble_path, ensemble_metadata = ensemble_predictions_fixed(
        temporal_stacked=temporal_stacked,
        spatial_stacked=spatial_stacked,
        ensemble_method="average"
    )
    
    print(f"\nEnsemble predictions complete!")
    print(f"Temporal ensemble: {temporal_ensemble_path}")
    print(f"Spatial ensemble: {spatial_ensemble_path}")
    print(f"Ensemble metadata: {ensemble_metadata}")
    
    # Verify the ensemble shapes
    print("\nVerifying ensemble prediction shapes...")
    with h5py.File(temporal_ensemble_path, 'r') as f:
        ensemble_shape = f['predictions'].shape
        print(f"Temporal ensemble shape: {ensemble_shape}")
        print(f"Expected format: (n_events=2, n_timepoints=5, n_samples={ensemble_shape[-1]})")
    
    print("\nFixed ensemble pipeline completed successfully!")
else:
    print("\nERROR: No prediction paths found in deployment details!")
    print("The deployment details may not contain the expected prediction paths.")
    
    # Let's use the original approach from run_stack_predictions_fixed.py
    print("\nTrying to use the stack_predictions step directly with deployment details...")
    
    # Run the stack predictions step with just deployment details
    print("\n=== Running Stack Predictions Step ===")
    temporal_stacked, spatial_stacked = stack_predictions(
        deployment_details=deployment_details,
        temporal_predictions_paths=[],  # Empty list
        spatial_predictions_paths=[]     # Empty list
    )
    
    print(f"\nStacking complete!")
    print(f"Temporal stacked models: {len(temporal_stacked)}")
    print(f"Spatial stacked models: {len(spatial_stacked)}")
    
    if temporal_stacked:
        # Show the shapes of stacked predictions
        print("\nStacked prediction shapes:")
        for i, (model_name, predictions) in enumerate(temporal_stacked.items()):
            if i < 5:
                print(f"  {model_name}: {predictions.shape}")
        
        # Continue with the FIXED ensemble predictions
        print("\n=== Running Fixed Ensemble Predictions Step ===")
        temporal_ensemble_path, spatial_ensemble_path, ensemble_metadata = ensemble_predictions_fixed(
            temporal_stacked=temporal_stacked,
            spatial_stacked=spatial_stacked,
            ensemble_method="average"
        )
        
        print(f"\nEnsemble predictions complete!")
        print(f"Temporal ensemble: {temporal_ensemble_path}")
        print(f"Spatial ensemble: {spatial_ensemble_path}")
        print(f"Ensemble metadata: {ensemble_metadata}")
        
        # Verify the ensemble shapes
        print("\nVerifying ensemble prediction shapes...")
        with h5py.File(temporal_ensemble_path, 'r') as f:
            ensemble_shape = f['predictions'].shape
            print(f"Temporal ensemble shape: {ensemble_shape}")
            print(f"Expected format: (n_events=2, n_timepoints=5, n_samples={ensemble_shape[-1]})")
        
        print("\nFixed ensemble pipeline completed successfully!")