import numpy as np
from src.evaluation_metrics import concordance_index_censored

# Create test data
np.random.seed(42)
n_samples = 1000

# Simulate times
times = np.random.exponential(scale=1000, size=n_samples)

# Simulate events (1 = event occurred, 0 = censored)
events = np.random.binomial(1, 0.7, size=n_samples)

# Simulate predictions where higher values correlate with events
predictions = np.random.beta(2, 5, size=n_samples)
# Add correlation: if event occurred, increase prediction
predictions[events == 1] += np.random.normal(0.3, 0.1, size=np.sum(events == 1))
predictions = np.clip(predictions, 0, 1)

# Calculate C-index
c_index = concordance_index_censored(
    events == 1,
    times,
    predictions
)[0]

print(f"Test C-index with positive predictions: {c_index:.4f}")
print(f"Expected range: [0.5, 1.0]")
print(f"Mean prediction for events: {predictions[events == 1].mean():.4f}")
print(f"Mean prediction for censored: {predictions[events == 0].mean():.4f}")

# Also test the full calculate_all_metrics function
from src.evaluation_metrics import calculate_all_metrics

# Create CIF-style predictions (2 events, 5 time points, n_samples)
cif_predictions = np.zeros((2, 5, n_samples))
time_points = np.array([365, 730, 1095, 1460, 1825])

# Simulate increasing CIF over time
for t in range(5):
    cif_predictions[0, t, :] = predictions * (t + 1) / 5  # Event 1
    cif_predictions[1, t, :] = (1 - predictions) * (t + 1) / 5  # Event 2 (inverse)

# Create competing risk events
cr_events = np.zeros(n_samples)
cr_events[events == 1] = np.random.choice([1, 2], size=np.sum(events == 1))

# Test full metrics
metrics = calculate_all_metrics(times, cr_events, cif_predictions, time_points)

print("\nFull metrics test:")
print(f"IBS: {metrics['ibs']:.4f}")
print(f"C-index Event 1: {metrics['cidx_event1']:.4f}")
print(f"C-index Event 2: {metrics['cidx_event2']:.4f}")
print(f"NLL: {metrics['nll']:.4f}")