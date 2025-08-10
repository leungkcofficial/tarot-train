"""
Script to run stack_predictions step using cached artifacts with correct paths
"""

from zenml.client import Client
from steps.stack_predictions import stack_predictions
from steps.ensemble_predictions import ensemble_predictions
from steps.ensemble_evaluator import ensemble_evaluator
import pandas as pd

print("Loading cached artifacts from process_models_sequentially step...")

# Load the three artifacts from the process_models_sequentially step
client = Client()

# Load deployment details (first output)
print("\nLoading deployment details...")
artifact1 = client.get_artifact_version("ea12d9e2-1225-4a5e-8c49-e8443047fe21")
all_deployment_details = artifact1.load()
print(f"Loaded {len(all_deployment_details)} deployment details")

# Extract the correct prediction paths from deployment details
print("\nExtracting prediction paths from deployment details...")
all_temporal_predictions = []
all_spatial_predictions = []

for details in all_deployment_details:
    temporal_path = details.get('temporal_test_predictions_path', '')
    spatial_path = details.get('spatial_test_predictions_path', '')
    all_temporal_predictions.append(temporal_path)
    all_spatial_predictions.append(spatial_path)

print(f"Extracted {len(all_temporal_predictions)} temporal prediction paths")
print(f"Extracted {len(all_spatial_predictions)} spatial prediction paths")

# Verify paths are not empty
non_empty_temporal = sum(1 for path in all_temporal_predictions if path)
non_empty_spatial = sum(1 for path in all_spatial_predictions if path)
print(f"Non-empty temporal paths: {non_empty_temporal}")
print(f"Non-empty spatial paths: {non_empty_spatial}")

# Now run the stack_predictions step
print("\n=== Running Stack Predictions Step ===")
temporal_stacked, spatial_stacked = stack_predictions(
    deployment_details=all_deployment_details,
    temporal_predictions_paths=all_temporal_predictions,
    spatial_predictions_paths=all_spatial_predictions
)

print(f"\nStacking complete!")
print(f"Temporal stacked models: {len(temporal_stacked)}")
print(f"Spatial stacked models: {len(spatial_stacked)}")

# Continue with ensemble predictions
print("\n=== Running Ensemble Predictions Step ===")
temporal_ensemble_path, spatial_ensemble_path, ensemble_metadata = ensemble_predictions(
    temporal_stacked=temporal_stacked,
    spatial_stacked=spatial_stacked,
    ensemble_method="average",
    output_dir="results/final_deploy/ensemble_predictions"
)

print(f"\nEnsemble predictions saved:")
print(f"Temporal: {temporal_ensemble_path}")
print(f"Spatial: {spatial_ensemble_path}")

# For evaluation, we need the original test dataframes
print("\n=== Note: Ensemble Evaluator Step ===")
print("To run the ensemble_evaluator step, you need to provide:")
print("1. temporal_test_df - Original temporal test dataframe with labels")
print("2. spatial_test_df - Original spatial test dataframe with labels")
print("\nThese should be the same dataframes used in the original pipeline.")

# Save the paths for later use
import json
output_info = {
    "temporal_ensemble_path": temporal_ensemble_path,
    "spatial_ensemble_path": spatial_ensemble_path,
    "ensemble_metadata": ensemble_metadata,
    "num_models": len(all_deployment_details),
    "timestamp": ensemble_metadata.get("timestamp", "")
}

with open("ensemble_output_info.json", "w") as f:
    json.dump(output_info, f, indent=2)

print(f"\nSaved ensemble output information to ensemble_output_info.json")
print("\n=== Script Complete ===")