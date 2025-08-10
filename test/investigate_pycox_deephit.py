#!/usr/bin/env python3
"""
Investigate PyCox DeepHit methods for competing risks predictions.
"""

import torch
import numpy as np
from pycox.models import DeepHit
from src.nn_architectures import LSTMDeepHit

def investigate_deephit_methods():
    """Investigate available DeepHit prediction methods."""
    
    print("=== PyCox DeepHit Methods Investigation ===\n")
    
    # Create a simple model for testing
    net = LSTMDeepHit(
        input_dim=10,
        sequence_length=8,
        lstm_hidden_dims=[32],
        output_dim=5,  # 5 time intervals
        num_causes=2,  # 2 competing events
        dropout=0.1,
        bidirectional=False
    )
    
    time_grid = np.array([365, 730, 1095, 1460, 1825])
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    model = DeepHit(net, optimizer=optimizer, alpha=0.2, sigma=0.1, duration_index=time_grid)
    
    print("1. Available DeepHit methods:")
    methods = [method for method in dir(model) if not method.startswith('_')]
    prediction_methods = [method for method in methods if 'predict' in method.lower()]
    
    for method in prediction_methods:
        print(f"   - {method}")
    
    print(f"\n2. All methods containing 'predict':")
    for method in prediction_methods:
        try:
            method_obj = getattr(model, method)
            if callable(method_obj):
                print(f"   - {method}: {method_obj.__doc__[:100] if method_obj.__doc__ else 'No docstring'}...")
        except Exception as e:
            print(f"   - {method}: Error accessing - {e}")
    
    # Test with dummy data
    batch_size = 5
    x_test = torch.randn(batch_size, 8, 10)  # (batch, sequence, features)
    
    print(f"\n3. Testing prediction methods with input shape: {x_test.shape}")
    
    # Test predict_surv_df (current method)
    try:
        with torch.no_grad():
            surv_df = model.predict_surv_df(x_test)
        print(f"   - predict_surv_df: {type(surv_df)}, shape: {surv_df.shape}")
        print(f"     Index (time points): {surv_df.index.tolist()}")
        print(f"     Columns (samples): {len(surv_df.columns)}")
    except Exception as e:
        print(f"   - predict_surv_df: Error - {e}")
    
    # Test other prediction methods
    other_methods = ['predict_cif', 'predict_pmf', 'predict']
    for method_name in other_methods:
        if hasattr(model, method_name):
            try:
                method = getattr(model, method_name)
                with torch.no_grad():
                    result = method(x_test)
                print(f"   - {method_name}: {type(result)}, shape: {result.shape if hasattr(result, 'shape') else 'No shape'}")
                
                # If it's a DataFrame, show more details
                if hasattr(result, 'index') and hasattr(result, 'columns'):
                    print(f"     Index: {result.index.tolist()}")
                    print(f"     Columns: {len(result.columns)}")
                    
                    # Check if this could be competing risks format
                    if len(result.index) == 10:  # 2 causes × 5 time points
                        print(f"     COMPETING RISKS FORMAT DETECTED!")
                        print(f"     - Rows 0-4 (Cause 1): {result.iloc[0:5, 0].values}")
                        print(f"     - Rows 5-9 (Cause 2): {result.iloc[5:10, 0].values}")
                        
            except Exception as e:
                print(f"   - {method_name}: Error - {e}")
        else:
            print(f"   - {method_name}: Method not available")
    
    print(f"\n4. Model configuration:")
    print(f"   - Duration index: {model.duration_index}")
    print(f"   - Number of causes: {getattr(model, 'num_causes', 'Unknown')}")
    
    # Check if there are cause-specific methods
    cause_methods = [method for method in methods if 'cause' in method.lower() or 'cif' in method.lower()]
    if cause_methods:
        print(f"\n5. Cause-specific methods found:")
        for method in cause_methods:
            print(f"   - {method}")
    else:
        print(f"\n5. No obvious cause-specific methods found")

if __name__ == "__main__":
    investigate_deephit_methods()