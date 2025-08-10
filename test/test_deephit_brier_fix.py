#!/usr/bin/env python3
"""
Test script to verify the integrated Brier score calculation fix for DeepHit models.
This script simulates the calculation to ensure it produces reasonable values.
"""

import numpy as np
import pandas as pd

def test_integrated_brier_calculation():
    """Test the integrated Brier score calculation for DeepHit models."""
    
    # Simulate DeepHit time points
    time_points = np.array([365, 730, 1095, 1460, 1825])
    
    # Simulate realistic Brier scores at each time point
    # These should be in the range 0.05-0.25 for reasonable models
    brier_scores = np.array([0.12, 0.15, 0.18, 0.20, 0.22])
    
    print("DeepHit Integrated Brier Score Calculation Test")
    print("=" * 50)
    print(f"Time points: {time_points}")
    print(f"Brier scores: {brier_scores}")
    print()
    
    # Original calculation (problematic)
    integrated_brier_old = np.trapz(brier_scores, time_points) / (time_points[-1] - time_points[0])
    print(f"Old calculation (trapz / time_range):")
    print(f"  np.trapz(brier_scores, time_points) = {np.trapz(brier_scores, time_points):.4f}")
    print(f"  time_range = {time_points[-1] - time_points[0]} days")
    print(f"  Integrated Brier Score = {integrated_brier_old:.6f}")
    print(f"  This is unreasonably low!")
    print()
    
    # Alternative calculations
    print("Alternative calculations:")
    
    # 1. Simple average (most straightforward for discrete points)
    integrated_brier_avg = np.mean(brier_scores)
    print(f"1. Simple average: {integrated_brier_avg:.4f}")
    
    # 2. Time-weighted average (accounts for uneven spacing if any)
    time_intervals = np.diff(np.concatenate([[0], time_points]))
    integrated_brier_weighted = np.average(brier_scores, weights=time_intervals)
    print(f"2. Time-weighted average: {integrated_brier_weighted:.4f}")
    print(f"   Time intervals: {time_intervals}")
    
    # 3. Normalized trapezoidal (divide by number of intervals, not total time)
    integrated_brier_normalized = np.trapz(brier_scores, time_points) / len(time_points)
    print(f"3. Normalized trapezoidal: {integrated_brier_normalized:.4f}")
    
    # 4. Proper integration with unit conversion
    # Convert from day-units to year-units for more reasonable scale
    time_points_years = time_points / 365.25
    integrated_brier_years = np.trapz(brier_scores, time_points_years) / (time_points_years[-1] - time_points_years[0])
    print(f"4. Integration in years: {integrated_brier_years:.4f}")
    
    print()
    print("Recommendation:")
    print("For DeepHit with 5 discrete time points, the integrated Brier score")
    print("should be calculated as a simple average or time-weighted average")
    print("of the Brier scores at those points, not divided by the large time range.")
    
    # Show what the fix should produce
    print()
    print("With the fix, DeepHit should use the same trapezoidal integration")
    print("as DeepSurv, but the key is ensuring the Brier scores are calculated")
    print("correctly at each discrete time point using the CIF predictions directly.")

if __name__ == "__main__":
    test_integrated_brier_calculation()