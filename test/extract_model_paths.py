import json
from pathlib import Path
from collections import defaultdict

def extract_model_paths(json_file=None):
    """Extract all model paths and prediction paths from the data.json file."""
    
    if json_file is None:
        # Use the ZenML artifact path
        json_file = '/home/goma/.config/zenml/local_stores/f53346f0-79a8-49e4-b4ed-1cfb14dfa3fb/process_models_sequentially/output_0/67c9050f-777c-40ad-b087-a4bd53d6c47a/99a24532/data.json'
    
    print(f"Loading data from: {json_file}")
    
    # Load the JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Initialize containers for paths
    model_paths = defaultdict(list)
    
    # Extract paths from each model
    for i, model_data in enumerate(data):
        model_name = model_data.get('model_name', f'Model_{i}')
        model_type = model_data.get('model_type', 'unknown')
        model_endpoint = model_data.get('model_endpoint', 'unknown')
        
        # Create a model identifier
        model_id = f"{model_name} ({model_type}, Event {model_endpoint})"
        
        # Extract the paths
        paths = {
            'original_model_path': model_data.get('original_model_path'),
            'temporal_test_predictions_path': model_data.get('temporal_test_predictions_path'),
            'temporal_test_metadata_path': model_data.get('temporal_test_metadata_path'),
            'spatial_test_predictions_path': model_data.get('spatial_test_predictions_path'),
            'spatial_test_metadata_path': model_data.get('spatial_test_metadata_path')
        }
        
        model_paths[model_id] = paths
    
    return model_paths, data

def display_paths(model_paths):
    """Display all extracted paths in an organized format."""
    
    print(f"Total models found: {len(model_paths)}")
    print("=" * 80)
    
    # Group by model type
    deepsurv_models = {k: v for k, v in model_paths.items() if 'deepsurv' in k.lower()}
    deephit_models = {k: v for k, v in model_paths.items() if 'deephit' in k.lower()}
    
    print(f"\nDeepSurv models: {len(deepsurv_models)}")
    print(f"DeepHit models: {len(deephit_models)}")
    print("=" * 80)
    
    # Display DeepSurv models
    print("\n### DEEPSURV MODELS ###")
    for i, (model_id, paths) in enumerate(sorted(deepsurv_models.items()), 1):
        print(f"\n{i}. {model_id}")
        for path_type, path_value in paths.items():
            print(f"   {path_type}: {path_value}")
    
    # Display DeepHit models
    print("\n### DEEPHIT MODELS ###")
    for i, (model_id, paths) in enumerate(sorted(deephit_models.items()), 1):
        print(f"\n{i}. {model_id}")
        for path_type, path_value in paths.items():
            print(f"   {path_type}: {path_value}")

def create_path_lists(model_paths):
    """Create separate lists for each type of path."""
    
    all_paths = {
        'original_model_paths': [],
        'temporal_test_predictions_paths': [],
        'temporal_test_metadata_paths': [],
        'spatial_test_predictions_paths': [],
        'spatial_test_metadata_paths': []
    }
    
    for model_id, paths in model_paths.items():
        all_paths['original_model_paths'].append(paths['original_model_path'])
        all_paths['temporal_test_predictions_paths'].append(paths['temporal_test_predictions_path'])
        all_paths['temporal_test_metadata_paths'].append(paths['temporal_test_metadata_path'])
        all_paths['spatial_test_predictions_paths'].append(paths['spatial_test_predictions_path'])
        all_paths['spatial_test_metadata_paths'].append(paths['spatial_test_metadata_path'])
    
    return all_paths

def verify_paths_exist(all_paths):
    """Verify which paths actually exist on disk."""
    
    print("\n### PATH VERIFICATION ###")
    
    for path_type, paths in all_paths.items():
        print(f"\n{path_type}:")
        existing = 0
        missing = 0
        
        for path in paths:
            if path and Path(path).exists():
                existing += 1
            else:
                missing += 1
                if path:
                    print(f"   MISSING: {path}")
        
        print(f"   Existing: {existing}, Missing: {missing}")

def main():
    """Main function to extract and display all model paths."""
    
    # Extract paths
    model_paths, raw_data = extract_model_paths()
    
    # Display organized paths
    display_paths(model_paths)
    
    # Create path lists
    all_paths = create_path_lists(model_paths)
    
    # Display summary
    print("\n### SUMMARY ###")
    for path_type, paths in all_paths.items():
        print(f"{path_type}: {len(paths)} paths")
    
    # Verify paths exist
    verify_paths_exist(all_paths)
    
    # Save path lists to separate files for easy access
    print("\n### SAVING PATH LISTS ###")
    for path_type, paths in all_paths.items():
        filename = f"{path_type}.txt"
        with open(filename, 'w') as f:
            for path in paths:
                f.write(f"{path}\n")
        print(f"Saved {len(paths)} paths to {filename}")
    
    # Also save as JSON for programmatic access
    with open('all_model_paths.json', 'w') as f:
        json.dump(all_paths, f, indent=2)
    print("\nAll paths saved to all_model_paths.json")
    
    return model_paths, all_paths

if __name__ == "__main__":
    model_paths, all_paths = main()