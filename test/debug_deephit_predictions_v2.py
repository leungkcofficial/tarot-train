import torch
import numpy as np
from pycox.models import DeepHitSingle
from src.nn_architectures import CNNMLP, LSTMDeepHit

# Test with a simple example
device = 'cpu'

# Create a simple DeepHit model for testing
# For ANN model
net_ann = CNNMLP(
    in_features=11,
    hidden_dims=[50, 30],
    out_features=5,  # 5 time intervals
    num_causes=2,
    dropout=0.1,
    batch_norm=True
)

# Create DeepHitSingle model
model_ann = DeepHitSingle(net_ann)

# Test with 2D input (ANN case)
print("Testing ANN model with 2D input:")
X_2d = torch.randn(10, 11)  # 10 samples, 11 features
try:
    # Check available methods
    print("Available prediction methods:", [m for m in dir(model_ann) if 'predict' in m])
    
    # Try predict_pmf
    if hasattr(model_ann, 'predict_pmf'):
        pmf = model_ann.predict_pmf(X_2d)
        print(f"PMF shape: {pmf.shape}")
        
        # Convert PMF to CIF
        # PMF is probability mass function, CIF is cumulative incidence function
        # CIF = cumsum(PMF) along time axis
        if isinstance(pmf, np.ndarray):
            cif = np.cumsum(pmf, axis=1)
            print(f"CIF shape: {cif.shape}")
    
except Exception as e:
    print(f"Error with 2D input: {e}")

# Test with 3D input (LSTM case)
print("\n\nTesting LSTM model with 3D input:")
net_lstm = LSTMDeepHit(
    input_dim=11,
    sequence_length=10,
    lstm_hidden_dims=[64, 64],
    output_dim=5,
    num_causes=2,
    dropout=0.1,
    bidirectional=False
)

model_lstm = DeepHitSingle(net_lstm)

X_3d = torch.randn(10, 10, 11)  # 10 samples, 10 time steps, 11 features
try:
    # Direct prediction will fail
    pmf = model_lstm.predict_pmf(X_3d)
    print(f"PMF shape: {pmf.shape}")
except Exception as e:
    print(f"Error with 3D input (expected): {e}")
    
    # Try manual forward pass
    print("\nTrying manual forward pass:")
    model_lstm.net.eval()
    with torch.no_grad():
        # Get raw output from network
        output = model_lstm.net(X_3d)
        print(f"Raw output shape: {output.shape}")
        
        # The output should be logits that need to be converted to probabilities
        # For DeepHit, output is typically (batch_size, n_intervals * n_causes)
        # We need to reshape and apply softmax
        
        n_intervals = 5
        n_causes = 2
        
        # Reshape output
        if len(output.shape) == 2:
            batch_size = output.shape[0]
            # Reshape to (batch_size, n_causes, n_intervals)
            output_reshaped = output.view(batch_size, n_causes, n_intervals)
            print(f"Reshaped output: {output_reshaped.shape}")
            
            # Apply softmax to get PMF
            # DeepHit uses a special transformation
            # We need to pad and apply softmax
            
            # This is a simplified version - actual implementation may differ
            pmf = torch.softmax(output_reshaped, dim=2)
            print(f"PMF shape after softmax: {pmf.shape}")
            
            # Convert to CIF
            cif = torch.cumsum(pmf, dim=2)
            print(f"CIF shape: {cif.shape}")
            
            # Convert to numpy and transpose to (n_causes, n_intervals, batch_size)
            cif_np = cif.detach().cpu().numpy()
            cif_final = cif_np.transpose(1, 2, 0)
            print(f"Final CIF shape: {cif_final.shape}")