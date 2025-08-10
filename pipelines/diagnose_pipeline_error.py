#!/usr/bin/env python3
"""
Diagnostic script to identify the exact location of the balancing method error in the pipeline.
"""

import os
import sys
import pandas as pd
import numpy as np
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_balancing_error():
    """Test the balancing method error with model 1 configuration."""
    
    # Load model 1 configuration
    config_path = "results/final_deploy/model_config/model_config.csv"
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return
        
    config_df = pd.read_csv(config_path)
    model_1_config = config_df[config_df['Model No.'] == 1].iloc[0]
    
    print("Model 1 Configuration:")
    print(f"  Algorithm: {model_1_config['Algorithm']}")
    print(f"  Structure: {model_1_config['Structure']}")
    print(f"  Balancing Method: {model_1_config['Balancing Method']}")
    print(f"  Type of Balancing: {type(model_1_config['Balancing Method'])}")
    print(f"  Is NaN: {pd.isna(model_1_config['Balancing Method'])}")
    print()
    
    # Test the error scenario
    balancing = model_1_config['Balancing Method']
    
    # Test 1: Direct 'in' operator (this will fail)
    print("Test 1: Direct 'in' operator")
    try:
        result = 'None' in balancing
        print(f"  Result: {result}")
    except TypeError as e:
        print(f"  ERROR: {e}")
        print("  This is the error!")
    
    # Test 2: String conversion first (this works)
    print("\nTest 2: String conversion first")
    try:
        result = 'None' in str(balancing)
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  ERROR: {e}")
    
    # Test 3: Check for NaN first (best practice)
    print("\nTest 3: Check for NaN first")
    try:
        if pd.isna(balancing):
            print("  Balancing is NaN, converting to 'None'")
            balancing = 'None'
        result = 'None' in balancing
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  ERROR: {e}")
    
    # Test which file is being used
    print("\n" + "="*60)
    print("Checking which pipeline file is being used...")
    
    # Check if final_deploy_v2.py has the problematic code
    v2_path = "pipelines/final_deploy_v2.py"
    if os.path.exists(v2_path):
        with open(v2_path, 'r') as f:
            content = f.read()
            if "'NearMiss' in balancing_method" in content:
                print(f"WARNING: {v2_path} contains problematic code!")
                print("  Found: 'NearMiss' in balancing_method")
            if "'NearMiss version' in balancing_method" in content:
                print(f"WARNING: {v2_path} contains problematic code!")
                print("  Found: 'NearMiss version' in balancing_method")
    
    # Check if final_deploy_v2_fixed.py has the fix
    fixed_path = "pipelines/final_deploy_v2_fixed.py"
    if os.path.exists(fixed_path):
        with open(fixed_path, 'r') as f:
            content = f.read()
            if "pd.isna(balancing_method)" in content:
                print(f"GOOD: {fixed_path} contains the fix!")
                print("  Found: pd.isna(balancing_method)")
            if "'NearMiss' in str(balancing_method)" in content:
                print(f"GOOD: {fixed_path} uses string conversion!")
                print("  Found: 'NearMiss' in str(balancing_method)")
    
    print("\n" + "="*60)
    print("RECOMMENDATION:")
    print("Make sure you're running pipelines/final_deploy_v2_fixed.py")
    print("NOT pipelines/final_deploy_v2.py")

if __name__ == "__main__":
    test_balancing_error()