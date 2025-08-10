"""
Script to run stack_predictions step using cached artifacts from process_models_sequentially
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

# Load temporal predictions paths (second output)
print("\nLoading temporal predictions paths...")
artifact2 = client.get_artifact_version("b57ff343-cf23-413d-9934-659ca6f7e64b")
all_temporal_predictions = artifact2.load()
print(f"Loaded {len(all_temporal_predictions)} temporal prediction paths")

# Load spatial predictions paths (third output)
print("\nLoading spatial predictions paths...")
artifact3 = client.get_artifact_version("a2bc1805-0030-4bbe-9a68-4713551d2f91")
all_spatial_predictions = artifact3.load()
print(f"Loaded {len(all_spatial_predictions)} spatial prediction paths")

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
# You'll need to provide these or load them from somewhere
print("\n=== Note: Ensemble Evaluator Step ===")
print("To run the ensemble_evaluator step, you need to provide:")
print("1. temporal_test_df - Original temporal test dataframe with labels")
print("2. spatial_test_df - Original spatial test dataframe with labels")
print("\nThese should be the same dataframes used in the original pipeline.")
print("If you have them saved or can load them, uncomment the code below:")

# # Example of how to run evaluation if you have the test dataframes
# evaluation_results = ensemble_evaluator(
#     temporal_ensemble_path=temporal_ensemble_path,
#     spatial_ensemble_path=spatial_ensemble_path,
#     ensemble_metadata=ensemble_metadata,
#     temporal_test_df=temporal_test_df,  # You need to load this
#     spatial_test_df=spatial_test_df,    # You need to load this
#     master_df_mapping_path="src/default_master_df_mapping.yml",
#     output_dir="results/final_deploy/ensemble_eval"
# )

print("\n=== Script Complete ===")
print("Stack predictions and ensemble predictions have been generated.")
print("To complete the evaluation, you'll need to provide the test dataframes.")