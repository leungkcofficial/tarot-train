import h5py
import json

def inspect_h5_structure():
    """Inspect the structure of H5 prediction files"""
    
    # Load the model data from JSON
    json_file = '/home/goma/.config/zenml/local_stores/f53346f0-79a8-49e4-b4ed-1cfb14dfa3fb/process_models_sequentially/output_0/67c9050f-777c-40ad-b087-a4bd53d6c47a/99a24532/data.json'
    with open(json_file, 'r') as f:
        model_data = json.load(f)
    
    # Get first DeepSurv model to inspect
    deepsurv_model = None
    deephit_model = None
    
    for model in model_data:
        if model['model_type'] == 'deepsurv' and deepsurv_model is None:
            deepsurv_model = model
        elif model['model_type'] == 'deephit' and deephit_model is None:
            deephit_model = model
        
        if deepsurv_model and deephit_model:
            break
    
    print("### INSPECTING H5 FILE STRUCTURE ###")
    print("=" * 80)
    
    # Inspect DeepSurv model
    if deepsurv_model:
        print(f"\nDeepSurv Model: {deepsurv_model['model_name']}")
        print(f"Temporal test file: {deepsurv_model['temporal_test_predictions_path']}")
        
        try:
            with h5py.File(deepsurv_model['temporal_test_predictions_path'], 'r') as f:
                print("\nKeys in DeepSurv temporal test H5 file:")
                for key in f.keys():
                    print(f"  - {key}: shape = {f[key].shape}, dtype = {f[key].dtype}")
                    
                # If there are nested groups, explore them
                for key in f.keys():
                    if isinstance(f[key], h5py.Group):
                        print(f"\n  Group '{key}' contains:")
                        for subkey in f[key].keys():
                            print(f"    - {subkey}: shape = {f[key][subkey].shape}, dtype = {f[key][subkey].dtype}")
        except Exception as e:
            print(f"Error reading file: {e}")
    
    # Inspect DeepHit model
    if deephit_model:
        print(f"\n\nDeepHit Model: {deephit_model['model_name']}")
        print(f"Temporal test file: {deephit_model['temporal_test_predictions_path']}")
        
        try:
            with h5py.File(deephit_model['temporal_test_predictions_path'], 'r') as f:
                print("\nKeys in DeepHit temporal test H5 file:")
                for key in f.keys():
                    print(f"  - {key}: shape = {f[key].shape}, dtype = {f[key].dtype}")
                    
                # If there are nested groups, explore them
                for key in f.keys():
                    if isinstance(f[key], h5py.Group):
                        print(f"\n  Group '{key}' contains:")
                        for subkey in f[key].keys():
                            print(f"    - {subkey}: shape = {f[key][subkey].shape}, dtype = {f[key][subkey].dtype}")
        except Exception as e:
            print(f"Error reading file: {e}")
    
    # Also check a few more models to see if structure is consistent
    print("\n\n### CHECKING MULTIPLE MODELS FOR CONSISTENCY ###")
    print("=" * 80)
    
    # Check first 3 DeepSurv and first 3 DeepHit models
    deepsurv_count = 0
    deephit_count = 0
    
    for model in model_data:
        if model['model_type'] == 'deepsurv' and deepsurv_count < 3:
            deepsurv_count += 1
            print(f"\n{model['model_name']}:")
            try:
                with h5py.File(model['temporal_test_predictions_path'], 'r') as f:
                    keys = list(f.keys())
                    print(f"  Keys: {keys}")
                    if 'predictions' in keys:
                        print(f"  predictions shape: {f['predictions'].shape}")
            except Exception as e:
                print(f"  Error: {e}")
        
        elif model['model_type'] == 'deephit' and deephit_count < 3:
            deephit_count += 1
            print(f"\n{model['model_name']}:")
            try:
                with h5py.File(model['temporal_test_predictions_path'], 'r') as f:
                    keys = list(f.keys())
                    print(f"  Keys: {keys}")
                    if 'predictions' in keys:
                        print(f"  predictions shape: {f['predictions'].shape}")
            except Exception as e:
                print(f"  Error: {e}")

if __name__ == "__main__":
    inspect_h5_structure()