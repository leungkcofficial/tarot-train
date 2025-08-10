"""
Load and process the cached artifacts directly from the file system.
"""

import os
import pickle
import numpy as np
import h5py
from steps.ensemble_predictions_fixed import ensemble_predictions_fixed

# Artifact paths from the 3-day run
artifact_paths = {
    'deployment_details': '/home/goma/.config/zenml/local_stores/f53346f0-79a8-49e4-b4ed-1cfb14dfa3fb/process_models_sequentially/output_0/67c9050f-777c-40ad-b087-a4bd53d6c47a/99a24532',
    'temporal_predictions': '/home/goma/.config/zenml/local_stores/f53346f0-79a8-49e4-b4ed-1cfb14dfa3fb/process_models_sequentially/output_1/67c9050f-777c-40ad-b087-a4bd53d6c47a/8784e2ea',
    'spatial_predictions': '/home/goma/.config/zenml/local_stores/f53346f0-79a8-49e4-b4ed-1cfb14dfa3fb/process_models_sequentially/output_2/67c9050f-777c-40ad-b087-a4bd53d6c47a/385bd4d0'
}

print("Loading cached artifacts from file system...")

# Load the artifacts
loaded_artifacts = {}
for name, path in artifact_paths.items():
    print(f"\nLoading {name} from {path}")
    
    # Try different file formats
    if os.path.exists(path):
        # Try pickle first
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                loaded_artifacts[name] = data
                print(f"  Loaded as pickle. Type: {type(data)}")
                if isinstance(data, list):
                    print(f"  List length: {len(data)}")
                    if data:
                        print(f"  First item type: {type(data[0])}")
        except:
            print("  Not a pickle file")
    else:
        # Check if it's a directory
        if os.path.isdir(path):
            print(f"  {path} is a directory")
            # List contents
            contents = os.listdir(path)
            print(f"  Contents: {contents}")
            
            # Try to load from subdirectories
            for item in contents:
                item_path = os.path.join(path, item)
                if os.path.isfile(item_path):
                    try:
                        with open(item_path, 'rb') as f:
                            data = pickle.load(f)
                            loaded_artifacts[name] = data
                            print(f"  Loaded from {item}. Type: {type(data)}")
                            break
                    except:
                        pass

# Check what we loaded
print("\n=== Loaded Artifacts Summary ===")
for name, data in loaded_artifacts.items():
    print(f"\n{name}:")
    if isinstance(data, list):
        print(f"  Type: list of {len(data)} items")
        if data:
            first_item = data[0]
            print(f"  First item type: {type(first_item)}")
            if isinstance(first_item, dict):
                print(f"  First item keys: {list(first_item.keys())[:5]}...")
            elif isinstance(first_item, str):
                print(f"  First item: {first_item}")
    elif isinstance(data, dict):
        print(f"  Type: dict with {len(data)} keys")
        print(f"  Keys: {list(data.keys())[:5]}...")

# Process based on what we found
if 'deployment_details' in loaded_artifacts:
    deployment_details = loaded_artifacts['deployment_details']
    
    # Check if it's the expected format
    if isinstance(deployment_details, list) and deployment_details:
        if isinstance(deployment_details[0], dict):
            print("\nDeployment details are in expected format (list of dicts)")
            
            # Extract prediction paths
            temporal_paths = []
            spatial_paths = []
            
            for detail in deployment_details:
                if 'temporal_predictions_path' in detail:
                    temporal_paths.append(detail['temporal_predictions_path'])
                if 'spatial_predictions_path' in detail:
                    spatial_paths.append(detail['spatial_predictions_path'])
            
            print(f"\nExtracted {len(temporal_paths)} temporal paths")
            print(f"Extracted {len(spatial_paths)} spatial paths")
            
            # Check if paths exist
            if temporal_paths:
                existing_paths = sum(1 for p in temporal_paths if os.path.exists(p))
                print(f"Existing temporal paths: {existing_paths}/{len(temporal_paths)}")
                
                # Show sample paths
                print("\nSample temporal paths:")
                for i, path in enumerate(temporal_paths[:3]):
                    print(f"  {i}: {path}")
                    if os.path.exists(path):
                        # Check file size and content
                        with h5py.File(path, 'r') as f:
                            if 'predictions' in f:
                                shape = f['predictions'].shape
                            elif 'cif' in f:
                                shape = f['cif'].shape
                            else:
                                shape = "Unknown"
                            print(f"     Shape: {shape}")

# If we have temporal and spatial predictions lists directly
if 'temporal_predictions' in loaded_artifacts and 'spatial_predictions' in loaded_artifacts:
    temporal_preds = loaded_artifacts['temporal_predictions']
    spatial_preds = loaded_artifacts['spatial_predictions']
    
    print("\n=== Direct Predictions Found ===")
    print(f"Temporal predictions: {type(temporal_preds)}")
    print(f"Spatial predictions: {type(spatial_preds)}")
    
    if isinstance(temporal_preds, list):
        print(f"\nTemporal predictions list has {len(temporal_preds)} items")
        if temporal_preds:
            first = temporal_preds[0]
            print(f"First item type: {type(first)}")
            if isinstance(first, str) and first.endswith('.h5'):
                print("List contains H5 file paths")
                # These are the prediction file paths!
                temporal_paths = temporal_preds
                spatial_paths = spatial_preds if isinstance(spatial_preds, list) else []
                
                print(f"\nFound {len(temporal_paths)} temporal prediction paths")
                print(f"Found {len(spatial_paths)} spatial prediction paths")
                
                # Now we can run the stacking step
                from steps.stack_predictions import stack_predictions
                
                print("\n=== Running Stack Predictions Step ===")
                # We need deployment details for stacking
                # Let's create dummy deployment details from the paths
                deployment_details_for_stacking = []
                
                for i, (temp_path, spat_path) in enumerate(zip(temporal_paths, spatial_paths)):
                    # Infer model type from filename or index
                    if i < 12:
                        # First 12 are DeepSurv groups
                        model_type = 'DeepSurv'
                        model_name = f'deepsurv_group_{i}'
                    else:
                        # Next 12 are DeepHit
                        model_type = 'DeepHit'
                        model_name = f'deephit_{i-12}'
                    
                    detail = {
                        'model_config': {
                            'algorithm': model_type,
                            'model_no': i + 1
                        },
                        'temporal_predictions_path': temp_path,
                        'spatial_predictions_path': spat_path,
                        'model_identifier': model_name
                    }
                    deployment_details_for_stacking.append(detail)
                
                # Run stacking
                temporal_stacked, spatial_stacked = stack_predictions(
                    deployment_details=deployment_details_for_stacking,
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
                
                # Continue with ensemble predictions
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
                
                print("\nPipeline completed successfully!")