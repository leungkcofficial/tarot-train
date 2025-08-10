#!/usr/bin/env python3
"""
Test script for the fixed final deployment pipeline.
"""

import os
import sys
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipelines.final_deploy_v2_fixed import multi_model_deploy_pipeline

def test_fixed_pipeline():
    """Test the fixed deployment pipeline."""
    print("=" * 80)
    print("Testing Fixed Final Deployment Pipeline")
    print("=" * 80)
    
    try:
        # Run the main pipeline
        print("\nRunning pipeline with all 36 models...")
        pipeline = multi_model_deploy_pipeline()
        # In ZenML, calling the pipeline function directly executes it
        
        print("\n" + "=" * 80)
        print("Pipeline completed successfully!")
        print("=" * 80)
        
        # Check output files
        output_dir = Path("results/final_deploy")
        if output_dir.exists():
            print("\nOutput files created:")
            for file in sorted(output_dir.glob("*.npz")):
                print(f"  - {file.name}")
            
            # Check predictions directory
            pred_dir = output_dir / "predictions"
            if pred_dir.exists():
                print(f"\nIndividual model predictions: {len(list(pred_dir.glob('*.npz')))} files")
            
            # Check ensemble directory
            ensemble_dirs = list(output_dir.glob("ensemble_*"))
            if ensemble_dirs:
                latest_ensemble = sorted(ensemble_dirs)[-1]
                print(f"\nLatest ensemble directory: {latest_ensemble.name}")
                for file in sorted(latest_ensemble.glob("*.npz")):
                    print(f"  - {file.name}")
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("ERROR: Pipeline failed!")
        print("=" * 80)
        print(f"\nError type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        
        # Try to identify which model caused the error
        error_msg = str(e)
        if "model_" in error_msg:
            import re
            model_match = re.search(r'model_(\d+)', error_msg)
            if model_match:
                model_no = int(model_match.group(1))
                print(f"\nError occurred while processing model {model_no}")
                
                # Load model config to get details
                import pandas as pd
                config_path = "results/final_deploy/model_config/model_config.csv"
                if os.path.exists(config_path):
                    config_df = pd.read_csv(config_path)
                    model_info = config_df[config_df['model_no'] == model_no]
                    if not model_info.empty:
                        print("\nModel details:")
                        for col in ['algorithm', 'structure', 'balancing_method', 'optimization_target', 'event']:
                            if col in model_info.columns:
                                print(f"  {col}: {model_info.iloc[0][col]}")

if __name__ == "__main__":
    test_fixed_pipeline()