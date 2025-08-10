#!/usr/bin/env python3
"""
Debug script to identify the exact location of the balancing method error.
"""

import pandas as pd
import numpy as np

# Test the error scenario
balancing = float('nan')  # This is what model 1 has

print("Testing balancing method error...")
print(f"balancing value: {balancing}")
print(f"Type of balancing: {type(balancing)}")
print(f"Is NaN: {pd.isna(balancing)}")

# This will cause the error
try:
    result = 'None' in balancing
    print(f"'None' in balancing: {result}")
except TypeError as e:
    print(f"ERROR: {e}")
    print("This is the error we're seeing!")

# The fix
print("\nTesting the fix...")
if pd.isna(balancing) or str(balancing).lower() == 'nan':
    balancing = 'None'
    print(f"Fixed balancing: {balancing}")

# Now this works
if 'None' in str(balancing):
    print("Now we can check if 'None' is in the string")

# Another potential error location
balancing2 = float('nan')
try:
    if 'NearMiss' in balancing2:
        print("This won't print")
except TypeError as e:
    print(f"\nAnother error location: {e}")
    
# The proper way to handle it
if 'NearMiss' in str(balancing2):
    print("This won't print either, but won't error")
else:
    print("Properly handled NaN balancing")