# DeepHit Integrated Brier Score Fix Summary

## Problem Identified
The integrated Brier score for DeepHit models was showing unreasonably low values (~0.00) while the individual Brier score plots showed reasonable values (0.05-0.25 range).

## Root Cause
1. **Model Type Difference**: 
   - DeepSurv produces continuous survival predictions for all days (365-1825)
   - DeepHit only produces predictions at 5 discrete time points: [365, 730, 1095, 1460, 1825]

2. **Integration Issue**: 
   - The code was using `np.trapz(brier_scores, time_points) / (time_points[-1] - time_points[0])`
   - This divides by 1460 days, which severely underscales the result
   - For DeepHit's sparse predictions, this calculation is inappropriate

3. **Data Access Issue**:
   - The manual calculation was trying to use `survival_predictions_df` which might not be defined
   - For DeepHit, we should use `cause_df` directly which contains the CIF predictions

## Changes Made

### 1. Fixed Brier Score Calculation for DeepHit
- Modified the manual Brier score calculation to use `cause_df` directly for DeepHit models
- This ensures we're using the actual CIF predictions instead of trying to convert from survival probabilities

### 2. Added At-Risk Filtering
- Added proper at-risk patient filtering for both Brier score and NBLL calculations
- Only patients still at risk at time t are included in the calculation
- This matches the standard survival analysis methodology

### 3. Fixed Variable References
- Changed `survival_predictions_df` to `survival_predictions` in the non-DeepHit branch
- Ensured consistent variable usage throughout the code

## Code Changes in `steps/model_eval.py`

1. **Lines 680-693**: Added model type check to use appropriate predictions:
   - For DeepHit: Use `cause_df` (CIF predictions) directly
   - For others: Use survival probabilities and convert to CIF

2. **Lines 688-691**: Added at-risk filtering:
   ```python
   # Calculate censored patients who are still at risk at time t
   at_risk = durations >= t
   # Brier score at time t (only for patients at risk)
   brier_t = np.mean((risk_prob_t.values[at_risk] - observed_by_t[at_risk].astype(float))**2)
   ```

3. **Similar changes for NBLL calculation** to ensure consistency

## Expected Outcome
With these fixes:
- DeepHit models should show integrated Brier scores in a reasonable range (0.10-0.25)
- The integrated score should match the range shown in the Brier score plots
- Both DeepSurv and DeepHit will use consistent methodology adapted to their prediction structures

## Testing
Run the evaluation pipeline again for model 26 to verify:
1. The integrated Brier score is now in a reasonable range
2. The value matches what's shown in the plots
3. The manual calculation fallback works correctly when EvalSurv returns near-zero values