import torch
import numpy as np
from pycox.models import DeepHit
from src.nn_architectures import CNNMLP

# Create a simple DeepHit model
net = CNNMLP(
    in_features=11,
    hidden_dims=[50, 30],
    out_features=5,
    num_causes=2,
    dropout=0.1,
    batch_norm=True
)

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
time_grid = np.array([365, 730, 1095, 1460, 1825])
model = DeepHit(net, optimizer=optimizer, alpha=0.2, sigma=0.1, duration_index=time_grid)

# Check available methods
print("Available DeepHit methods:")
methods = [m for m in dir(model) if 'predict' in m and not m.startswith('_')]
for method in methods:
    print(f"  - {method}")

# Test with 2D input
X = torch.randn(10, 11)
print(f"\nInput shape: {X.shape}")

# Try different prediction methods
try:
    if hasattr(model, 'predict_cif'):
        print("\nTrying predict_cif...")
        cif = model.predict_cif(X)
        print(f"CIF shape: {cif.shape if hasattr(cif, 'shape') else type(cif)}")
except Exception as e:
    print(f"predict_cif error: {e}")

try:
    if hasattr(model, 'predict_surv'):
        print("\nTrying predict_surv...")
        surv = model.predict_surv(X)
        print(f"Surv type: {type(surv)}")
        if hasattr(surv, 'shape'):
            print(f"Surv shape: {surv.shape}")
        elif isinstance(surv, tuple):
            print(f"Surv is tuple with {len(surv)} elements")
            for i, s in enumerate(surv):
                print(f"  Element {i} shape: {s.shape if hasattr(s, 'shape') else type(s)}")
except Exception as e:
    print(f"predict_surv error: {e}")

try:
    if hasattr(model, 'predict_surv_df'):
        print("\nTrying predict_surv_df...")
        surv_df = model.predict_surv_df(X)
        print(f"Surv_df type: {type(surv_df)}")
        if hasattr(surv_df, 'shape'):
            print(f"Surv_df shape: {surv_df.shape}")
except Exception as e:
    print(f"predict_surv_df error: {e}")

try:
    if hasattr(model, 'predict_pmf'):
        print("\nTrying predict_pmf...")
        pmf = model.predict_pmf(X)
        print(f"PMF shape: {pmf.shape if hasattr(pmf, 'shape') else type(pmf)}")
except Exception as e:
    print(f"predict_pmf error: {e}")

# Check if model has special attributes for competing risks
print("\nModel attributes related to competing risks:")
if hasattr(model, 'num_causes'):
    print(f"  num_causes: {model.num_causes}")
if hasattr(model, 'duration_index'):
    print(f"  duration_index: {model.duration_index}")