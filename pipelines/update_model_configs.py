"""
Script to update model JSON configuration files with balance and optimization settings.
This is optional - the final_deploy_v2.py already extracts these from the CSV.
"""

import json
import os
import pandas as pd
from typing import Dict, Any


def get_balance_config(balancing_method: Any) -> Dict[str, Any]:
    """Convert balancing method string from CSV to configuration dict."""
    # Handle NaN/None values
    if pd.isna(balancing_method) or balancing_method == 'None':
        return {
            'enable': False,
            'method': None,
            'sampling_strategy': 'majority'
        }
    elif isinstance(balancing_method, str) and 'NearMiss version' in balancing_method:
        version = int(balancing_method.split()[-1])
        return {
            'enable': True,
            'method': 'near_miss',
            'sampling_strategy': 'majority',
            'near_miss_version': version
        }
    elif isinstance(balancing_method, str) and balancing_method == 'KNN':
        return {
            'enable': True,
            'method': 'knn',
            'sampling_strategy': 'majority'
        }
    else:
        return {
            'enable': True,
            'method': 'random_under_sampler',
            'sampling_strategy': 'majority'
        }


def get_optimization_config(optimization_target: str) -> Dict[str, Any]:
    """Convert optimization target string from CSV to configuration dict."""
    return {
        'metric': 'cidx' if 'Concordance' in optimization_target else 'loglik',
        'n_trials': 50,  # This was used during hyperparameter optimization
        'patience': 10,
        'seed': 42
    }


def update_json_configs(
    csv_path: str = "results/final_deploy/model_config/model_config.csv",
    config_dir: str = "results/final_deploy/model_config",
    update_files: bool = False
):
    """
    Update model JSON files with balance and optimization settings from CSV.
    
    Args:
        csv_path: Path to model_config.csv
        config_dir: Directory containing model JSON files
        update_files: If True, actually update the files. If False, just show what would be added.
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    
    for _, row in df.iterrows():
        model_no = row['Model No.']
        
        # Find the corresponding JSON file
        json_files = [f for f in os.listdir(config_dir) if f.startswith(f"model{model_no}_details_")]
        
        if not json_files:
            print(f"No JSON file found for model {model_no}")
            continue
            
        json_path = os.path.join(config_dir, json_files[0])
        
        # Load existing JSON
        with open(json_path, 'r') as f:
            config = json.load(f)
        
        # Add balance and optimization settings
        balance_config = get_balance_config(row['Balancing Method'])
        optimization_config = get_optimization_config(row['Optimization target'])
        
        # Show what would be added
        print(f"\nModel {model_no} ({row['Algorithm']} - {row['Structure']} - {row['Prediction Endpoint']}):")
        print(f"  Current JSON: {json_files[0]}")
        print(f"  Balance config to add: {balance_config}")
        print(f"  Optimization config to add: {optimization_config}")
        
        if update_files:
            # Add to config
            config['balance'] = balance_config
            config['optimization'] = optimization_config
            
            # Save updated JSON
            with open(json_path, 'w') as f:
                json.dump(config, f, indent=4)
            print(f"  ✓ Updated {json_path}")


def show_example_updated_json():
    """Show an example of what an updated JSON file would look like."""
    print("\n" + "="*60)
    print("EXAMPLE: Updated JSON file structure")
    print("="*60)
    
    example = {
        "model_type": "deepsurv",
        "input_dim": 11,
        "hidden_dims": [119, 124, 14],
        "output_dim": 1,
        "dropout": 0.03022661174579847,
        "time_grid": [365, 730, 1095, 1460, 1825],
        "alpha": None,
        "sigma": None,
        "model_path": "/mnt/dump/yard/projects/tarot2/results/model_details/model_weights.pt",
        "timestamp": "2025-07-06 08:46:23",
        "balance": {
            "enable": True,
            "method": "near_miss",
            "sampling_strategy": "majority",
            "near_miss_version": 1
        },
        "optimization": {
            "metric": "cidx",
            "n_trials": 50,
            "patience": 10,
            "seed": 42
        }
    }
    
    print(json.dumps(example, indent=4))


if __name__ == "__main__":
    print("Model Configuration Update Script")
    print("="*60)
    print("\nThis script can add balance and optimization settings to your model JSON files.")
    print("Note: This is OPTIONAL - final_deploy_v2.py already extracts these from the CSV.")
    print("\nOptions:")
    print("1. The current approach (recommended): Let final_deploy_v2.py extract from CSV")
    print("2. Add settings to JSON files: Run this script with update_files=True")
    
    # Show what would be added (dry run)
    print("\n" + "="*60)
    print("DRY RUN - Showing what would be added to each JSON file:")
    print("="*60)
    update_json_configs(update_files=False)
    
    # Show example
    show_example_updated_json()
    
    print("\n" + "="*60)
    print("To actually update the files, modify this script and set update_files=True")
    print("="*60)