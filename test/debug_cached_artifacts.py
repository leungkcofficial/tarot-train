"""
Debug script to check the contents of cached artifacts
"""

from zenml.client import Client
import json

print("Loading cached artifacts to debug...")

# Load the three artifacts
client = Client()

# Load deployment details
print("\n=== Deployment Details ===")
artifact1 = client.get_artifact_version("ea12d9e2-1225-4a5e-8c49-e8443047fe21")
deployment_details = artifact1.load()
print(f"Number of deployment details: {len(deployment_details)}")
if deployment_details:
    print("\nFirst deployment detail sample:")
    first = deployment_details[0]
    print(f"Keys: {list(first.keys())}")
    if 'model_config' in first:
        print(f"Model config: {first['model_config'].get('model_no', 'N/A')} - {first['model_config'].get('algorithm', 'N/A')}")
    if 'temporal_predictions_path' in first:
        print(f"Temporal predictions path: {first['temporal_predictions_path']}")
    if 'spatial_predictions_path' in first:
        print(f"Spatial predictions path: {first['spatial_predictions_path']}")

# Load temporal predictions paths
print("\n=== Temporal Predictions Paths ===")
artifact2 = client.get_artifact_version("b57ff343-cf23-413d-9934-659ca6f7e64b")
temporal_paths = artifact2.load()
print(f"Number of temporal paths: {len(temporal_paths)}")
print("First 5 temporal paths:")
for i, path in enumerate(temporal_paths[:5]):
    print(f"  {i+1}: {path}")

# Load spatial predictions paths
print("\n=== Spatial Predictions Paths ===")
artifact3 = client.get_artifact_version("a2bc1805-0030-4bbe-9a68-4713551d2f91")
spatial_paths = artifact3.load()
print(f"Number of spatial paths: {len(spatial_paths)}")
print("First 5 spatial paths:")
for i, path in enumerate(spatial_paths[:5]):
    print(f"  {i+1}: {path}")

# Check if paths are empty strings
print("\n=== Path Analysis ===")
empty_temporal = sum(1 for path in temporal_paths if not path or path == '')
empty_spatial = sum(1 for path in spatial_paths if not path or path == '')
print(f"Empty temporal paths: {empty_temporal}")
print(f"Empty spatial paths: {empty_spatial}")

# Check model configurations in deployment details
print("\n=== Model Configurations in Deployment Details ===")
for i, details in enumerate(deployment_details[:5]):
    config = details.get('model_config', {})
    print(f"\nModel {i+1}:")
    print(f"  Model No: {config.get('model_no', 'N/A')}")
    print(f"  Algorithm: {config.get('algorithm', 'N/A')}")
    print(f"  Structure: {config.get('structure', 'N/A')}")
    print(f"  Endpoint: {config.get('prediction_endpoint', 'N/A')}")