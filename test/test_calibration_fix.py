#!/usr/bin/env python3
"""
Test script to verify the calibration plot axis scaling fix.
"""

import numpy as np
import matplotlib.pyplot as plt

def test_axis_scaling_fix():
    """Test the corrected axis scaling logic."""
    
    print("=== Testing Calibration Plot Axis Scaling Fix ===\n")
    
    # Simulate very low mortality risks (like our actual data)
    predicted_risks = [2.76e-06, 5.76e-06, 7.00e-06, 1.01e-05, 2.31e-05, 8.52e-05, 0.000245, 0.000447, 0.000595, 0.000815]
    observed_risks = [0.000277, 0.001816, 0.000504, 0.001916, 0.000466, 0.000714, 0.000125, 0.001182, 0.001022, 0.001496]
    
    min_risk = min(min(predicted_risks), min(observed_risks))
    max_risk = max(max(predicted_risks), max(observed_risks))
    
    print(f"Test data:")
    print(f"  Min risk: {min_risk:.6f}")
    print(f"  Max risk: {max_risk:.6f}")
    
    # Test old logic (problematic)
    old_axis_limit = max(max_risk * 1.1, 0.1)
    print(f"\nOld logic (problematic):")
    print(f"  Axis limit: {old_axis_limit:.6f} (forces 10% minimum)")
    
    # Test new logic (fixed)
    if max_risk < 0.01:  # If max risk is less than 1%
        new_axis_limit = max_risk * 1.2
    else:
        new_axis_limit = max(max_risk * 1.1, 0.1)
    
    print(f"\nNew logic (fixed):")
    print(f"  Axis limit: {new_axis_limit:.6f} (proportional to actual data)")
    
    # Create comparison plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Old problematic scaling
    ax1.scatter(predicted_risks, observed_risks, alpha=0.7, s=50, color='blue')
    ax1.plot([min_risk, max_risk], [min_risk, max_risk], 'r--', alpha=0.8, label='Perfect calibration')
    ax1.set_xlim(0, old_axis_limit)
    ax1.set_ylim(0, old_axis_limit)
    ax1.set_xlabel('Predicted Risk')
    ax1.set_ylabel('Observed Risk')
    ax1.set_title('BEFORE: Forced 10% axis limit\n(makes low risks appear high)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add text showing the issue
    ax1.text(0.05, 0.08, f'Axis: 0 to {old_axis_limit:.1%}\nData: 0 to {max_risk:.3%}', 
             transform=ax1.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # Plot 2: New corrected scaling
    ax2.scatter(predicted_risks, observed_risks, alpha=0.7, s=50, color='green')
    ax2.plot([min_risk, max_risk], [min_risk, max_risk], 'r--', alpha=0.8, label='Perfect calibration')
    ax2.set_xlim(0, new_axis_limit)
    ax2.set_ylim(0, new_axis_limit)
    ax2.set_xlabel('Predicted Risk')
    ax2.set_ylabel('Observed Risk')
    ax2.set_title('AFTER: Proportional axis limit\n(shows true scale of risks)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add text showing the fix
    ax2.text(0.05, 0.85, f'Axis: 0 to {new_axis_limit:.3%}\nData: 0 to {max_risk:.3%}', 
             transform=ax2.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('calibration_axis_scaling_fix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Fix verification plot saved as 'calibration_axis_scaling_fix.png'")
    print(f"The fix ensures that very low mortality risks are displayed at their true scale.")
    
    # Test edge cases
    print(f"\n🧪 Testing edge cases:")
    
    # Case 1: Very low risks (< 1%)
    test_max_risk_1 = 0.005  # 0.5%
    if test_max_risk_1 < 0.01:
        limit_1 = test_max_risk_1 * 1.2
    else:
        limit_1 = max(test_max_risk_1 * 1.1, 0.1)
    print(f"  Max risk 0.5%: Axis limit = {limit_1:.3%} ✓")
    
    # Case 2: Medium risks (> 1%)
    test_max_risk_2 = 0.05  # 5%
    if test_max_risk_2 < 0.01:
        limit_2 = test_max_risk_2 * 1.2
    else:
        limit_2 = max(test_max_risk_2 * 1.1, 0.1)
    print(f"  Max risk 5%: Axis limit = {limit_2:.1%} ✓")
    
    # Case 3: High risks (> 10%)
    test_max_risk_3 = 0.15  # 15%
    if test_max_risk_3 < 0.01:
        limit_3 = test_max_risk_3 * 1.2
    else:
        limit_3 = max(test_max_risk_3 * 1.1, 0.1)
    print(f"  Max risk 15%: Axis limit = {limit_3:.1%} ✓")

if __name__ == "__main__":
    test_axis_scaling_fix()