#!/usr/bin/env python3
"""
Debug script to examine DeepHit prediction shapes and understand competing risks structure.
"""

import torch
import numpy as np
import pandas as pd
from pycox.models import DeepHit
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from src.nn_architectures import LSTMDeepHit

def debug_deephit_predictions():
    """Debug DeepHit prediction shapes for competing risks."""
    
    print("=== DeepHit Competing Risks Prediction Shape Analysis ===\n")
    
    # Simulate some test data
    batch_size = 10
    sequence_length = 8
    input_dim = 50
    num_time_intervals = 5
    num_causes = 2
    
    # Set device
    device = torch.device('cpu')  # Use CPU for debugging to avoid device issues
    
    # Create dummy input data
    x_test = torch.randn(batch_size, sequence_length, input_dim).to(device)
    
    # Create LSTM DeepHit model
    net = LSTMDeepHit(
        input_dim=input_dim,
        sequence_length=sequence_length,
        lstm_hidden_dims=[64],
        output_dim=num_time_intervals,
        num_causes=num_causes,
        dropout=0.2,
        bidirectional=True
    ).to(device)
    
    # Create time grid
    time_grid = np.array([365, 730, 1095, 1460, 1825])
    
    # Create DeepHit model
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    model = DeepHit(net, optimizer=optimizer, alpha=0.2, sigma=0.1, duration_index=time_grid)
    
    print(f"1. Input shape: {x_test.shape}")
    print(f"   - batch_size: {batch_size}")
    print(f"   - sequence_length: {sequence_length}")
    print(f"   - input_dim: {input_dim}")
    
    # Test raw network output
    with torch.no_grad():
        raw_output = net(x_test)
    
    print(f"\n2. Raw network output shape: {raw_output.shape}")
    print(f"   - Expected: (batch_size={batch_size}, num_causes={num_causes}, num_time_intervals={num_time_intervals})")
    print(f"   - Actual: {raw_output.shape}")
    
    # Test model prediction
    with torch.no_grad():
        predictions = model.predict_surv_df(x_test)
    
    print(f"\n3. Model prediction shape: {predictions.shape}")
    print(f"   - Type: {type(predictions)}")
    
    if isinstance(predictions, pd.DataFrame):
        print(f"   - DataFrame index (time points): {len(predictions.index)}")
        print(f"   - DataFrame columns (samples): {len(predictions.columns)}")
        print(f"   - Index values: {predictions.index.tolist()}")
        
        # Check if this is competing risks format
        if len(predictions.index) == num_time_intervals * num_causes:
            print(f"\n4. COMPETING RISKS DETECTED!")
            print(f"   - Total rows: {len(predictions.index)} = {num_time_intervals} time intervals × {num_causes} causes")
            print(f"   - This suggests predictions are stacked: [cause1_t1, cause1_t2, ..., cause2_t1, cause2_t2, ...]")
            
            # Try to separate the causes
            cause1_predictions = predictions.iloc[:num_time_intervals, :]
            cause2_predictions = predictions.iloc[num_time_intervals:, :]
            
            print(f"\n5. Separated predictions:")
            print(f"   - Cause 1 (Event 1) shape: {cause1_predictions.shape}")
            print(f"   - Cause 2 (Event 2) shape: {cause2_predictions.shape}")
            
            print(f"\n6. Sample values for first patient:")
            print(f"   - Cause 1 survival probabilities: {cause1_predictions.iloc[:, 0].values}")
            print(f"   - Cause 2 survival probabilities: {cause2_predictions.iloc[:, 0].values}")
            
        else:
            print(f"\n4. SINGLE EVENT FORMAT")
            print(f"   - Rows: {len(predictions.index)} (should equal {num_time_intervals})")
    
    print(f"\n7. Time grid: {time_grid}")
    print(f"   - Number of time points: {len(time_grid)}")
    
    return predictions, time_grid, num_causes

if __name__ == "__main__":
    predictions, time_grid, num_causes = debug_deephit_predictions()