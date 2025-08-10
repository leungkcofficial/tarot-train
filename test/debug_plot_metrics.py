#!/usr/bin/env python3

import sys
import os
sys.path.append('.')

# Test the plot_metrics_by_time function with sample data
from src.metric_calculator import plot_metrics_by_time

# Create sample metrics data that matches what should be passed from model_eval.py
sample_metrics = {
    'c_index': {
        'mean': 0.876,
        'lower': 0.850,
        'upper': 0.902
    },
    'integrated_brier_score': {
        'mean': 0.281,
        'lower': 0.261,
        'upper': 0.301
    },
    'integrated_nbll': {
        'mean': 1.266,
        'lower': 1.216,
        'upper': 1.316
    },
    'metrics_by_horizon': {
        '365': {
            'brier_score': {'mean': 0.25, 'lower': 0.20, 'upper': 0.30},
            'c_index': {'mean': 0.85, 'lower': 0.80, 'upper': 0.90}
        },
        '730': {
            'brier_score': {'mean': 0.28, 'lower': 0.23, 'upper': 0.33},
            'c_index': {'mean': 0.82, 'lower': 0.77, 'upper': 0.87}
        },
        '1095': {
            'brier_score': {'mean': 0.30, 'lower': 0.25, 'upper': 0.35},
            'c_index': {'mean': 0.80, 'lower': 0.75, 'upper': 0.85}
        }
    }
}

print("Testing plot_metrics_by_time function...")
print("Sample metrics structure:")
for key, value in sample_metrics.items():
    print(f"  {key}: {value}")

print("\nCalling plot_metrics_by_time...")
try:
    result = plot_metrics_by_time(sample_metrics, time_horizons=[365, 730, 1095], output_path="test_plot.png")
    print(f"Plot created successfully: {result}")
except Exception as e:
    print(f"Error creating plot: {e}")
    import traceback
    traceback.print_exc()

# Test with missing integrated metrics (simulating the problem)
print("\n" + "="*50)
print("Testing with missing integrated metrics...")

sample_metrics_missing = {
    'c_index': {
        'mean': 0.876,
        'lower': 0.850,
        'upper': 0.902
    },
    'metrics_by_horizon': {
        '365': {
            'brier_score': {'mean': 0.25, 'lower': 0.20, 'upper': 0.30},
            'c_index': {'mean': 0.85, 'lower': 0.80, 'upper': 0.90}
        }
    }
}

try:
    result = plot_metrics_by_time(sample_metrics_missing, time_horizons=[365], output_path="test_plot_missing.png")
    print(f"Plot with missing metrics created successfully: {result}")
except Exception as e:
    print(f"Error creating plot with missing metrics: {e}")
    import traceback
    traceback.print_exc()

# Test with float values instead of dict structure
print("\n" + "="*50)
print("Testing with float values instead of dict structure...")

sample_metrics_float = {
    'c_index': 0.876,
    'integrated_brier_score': 0.281,
    'integrated_nbll': 1.266,
    'metrics_by_horizon': {
        '365': {
            'brier_score': {'mean': 0.25, 'lower': 0.20, 'upper': 0.30},
            'c_index': {'mean': 0.85, 'lower': 0.80, 'upper': 0.90}
        }
    }
}

try:
    result = plot_metrics_by_time(sample_metrics_float, time_horizons=[365], output_path="test_plot_float.png")
    print(f"Plot with float values created successfully: {result}")
except Exception as e:
    print(f"Error creating plot with float values: {e}")
    import traceback
    traceback.print_exc()