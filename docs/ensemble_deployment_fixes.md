# Ensemble Deployment Pipeline Fixes

## Issues Resolved

### 1. ZenML StepArtifact Access Issue
**Problem**: Cannot use `len()` on ZenML StepArtifact objects within the pipeline definition.

**Solution**: 
- Modified `load_model_configurations` step to save configurations to a JSON file
- Created `process_models_sequentially` step that reads from the JSON file
- This avoids direct manipulation of ZenML artifacts in the pipeline

### 2. KNN Balancing Method Mapping
**Problem**: 'KNN' balancing method was incorrectly mapped to 'knn', which is not supported.

**Solution**: 
- Updated `model_config_utils.py` to map 'KNN' to 'enn' (Edited Nearest Neighbors)
- This aligns with the supported balancing methods in the system

## Updated Architecture

### Step Flow:
1. `load_model_configurations` - Loads configs and saves to JSON
2. `process_models_sequentially` - Reads JSON and processes models one by one
3. `stack_predictions` - Groups DeepSurv predictions (to be tested)
4. `ensemble_predictions` - Creates ensemble (to be tested)
5. `ensemble_evaluator` - Evaluates performance (to be tested)

### Key Changes:
- Added JSON file intermediary for model configurations
- Separated configuration loading from model processing
- Fixed balancing method mapping

## Next Steps
1. Continue testing with all 36 models
2. Enable and test the remaining steps (3-5)
3. Monitor memory usage during full pipeline execution
4. Validate ensemble predictions accuracy