"""
Fixed script to run stack predictions and ensemble from cached artifacts.
This version uses the fixed ensemble predictions step that handles different sample sizes.
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
temporal_predictions_artifact_id = "eeaa0e6f-f980-4575-b045-0d39e6bf5dc2"
spatial_predictions_artifact_id = "e5c5e5f5-5e5e-5e5e-5e5e-5e5e5e5e5e5e"  # Update this if needed

# Load the deployment details artifact
deployment_details_artifact = client.get_artifact_version(deployment_details_artifact_id)
deployment_details = deployment_details_artifact.load()

print(f"\nLoaded {len(deployment_details)} deployment details")

# Extract the correct prediction paths from deployment details
temporal_paths = []
spatial_paths = []

for detail in deployment_details:
    if 'temporal_predictions_path' in detail and detail['temporal_predictions_path']:
        temporal_paths.append(detail['temporal_predictions_path'])
    if 'spatial_predictions_path' in detail and detail['spatial_predictions_path']:
        spatial_paths.append(detail['spatial_predictions_path'])

print(f"\nExtracted prediction paths from deployment details:")
print(f"Temporal paths: {len(temporal_paths)}")
print(f"Spatial paths: {len(spatial_paths)}")

# Show sample counts for each model
print("\nChecking sample counts in prediction files...")
import h5py
import numpy as np

for i, (detail, temp_path) in enumerate(zip(deployment_details[:5], temporal_paths[:5])):
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
for model_name, predictions in list(temporal_stacked.items())[:5]:
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
print("\nNext steps:")
print("1. Run ensemble evaluation with the original test dataframes")
print("2. The ensemble predictions now have consistent shapes across all models")