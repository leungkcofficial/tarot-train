"""
Check the structure of H5 prediction files to understand their format.
"""

import h5py
import os


def check_h5_structure():
    """Check the structure of a sample H5 file."""
    
    predictions_dir = "results/final_deploy/individual_predictions"
    
    # Find a sample H5 file
    h5_files = [f for f in os.listdir(predictions_dir) if f.endswith('.h5')]
    
    if not h5_files:
        print("No H5 files found!")
        return
    
    # Check the first few files
    for i, filename in enumerate(h5_files[:5]):
        print(f"\n{'='*60}")
        print(f"File: {filename}")
        print(f"{'='*60}")
        
        filepath = os.path.join(predictions_dir, filename)
        
        try:
            with h5py.File(filepath, 'r') as f:
                print("Keys in file:")
                for key in f.keys():
                    print(f"  - {key}")
                    if isinstance(f[key], h5py.Dataset):
                        print(f"    Shape: {f[key].shape}")
                        print(f"    Dtype: {f[key].dtype}")
                
                # Try to access common keys
                if 'predictions' in f:
                    print(f"\nPredictions shape: {f['predictions'].shape}")
                if 'cif' in f:
                    print(f"\nCIF shape: {f['cif'].shape}")
                if 'survival_function' in f:
                    print(f"\nSurvival function shape: {f['survival_function'].shape}")
                    
        except Exception as e:
            print(f"Error reading file: {e}")


if __name__ == "__main__":
    check_h5_structure()