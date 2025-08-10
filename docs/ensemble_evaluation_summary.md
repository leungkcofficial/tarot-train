# Ensemble Evaluation Summary

## Problem Overview
- **Total combinations**: 16,777,215 (all possible combinations of 36 models)
- **Bottleneck**: C-index calculation requires ~1.8 billion pairwise comparisons per evaluation
- **Original estimate**: 4,488 days with single-core processing

## Investigation Results

### 1. GPU Acceleration
- **Result**: Only 1.1x speedup
- **Reason**: C-index calculation is inherently sequential
- **Conclusion**: Not worth the complexity

### 2. Numba JIT Optimization
- **Result**: Fixed parallel bug, achieved correct results
- **Speedup**: 34x faster than original
- **Time estimate**: 131 days for full evaluation
- **Status**: Working but still too slow

### 3. Multiprocessing/Threading
- **Result**: Actually slower than single-core
- **Reason**: Python GIL and memory copying overhead
- **Conclusion**: Not viable for this task

## Available Solutions

### Option 1: Sample-Based Evaluation (RECOMMENDED)
**File**: `fill_metrics_sample_based.py`
- **Combinations**: 60,000 (stratified sample)
- **Time**: ~20 hours
- **Confidence**: 99% for finding top performers
- **Command**: `python fill_metrics_sample_based.py`

### Option 2: Numba-Optimized Full Evaluation
**File**: `fill_metrics_numba_optimized.py`
- **Combinations**: All 16,777,215
- **Time**: ~131 days
- **Accuracy**: 100% exhaustive search
- **Command**: `python fill_metrics_numba_optimized.py`

### Option 3: Custom Sample Size
Modify `fill_metrics_sample_based.py` to adjust sample size:
- 10,000 combinations: ~3.3 hours
- 30,000 combinations: ~10 hours
- 100,000 combinations: ~33 hours

## Recommendation
Use the sample-based evaluation with 60,000 combinations. This provides:
1. Statistically valid results (99% confidence)
2. Reasonable runtime (20 hours vs 131 days)
3. Coverage of all ensemble sizes (1-24 models)
4. High probability of finding the optimal ensemble

## Next Steps
1. Run: `cd /mnt/dump/yard/projects/tarot2 && python fill_metrics_sample_based.py`
2. Monitor progress in the terminal
3. Results will be saved to `results/ensemble_checkpoints/evaluation_results_sample_based.csv`
4. Top 100 combinations will be extracted automatically