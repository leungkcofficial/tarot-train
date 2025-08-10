"""
Final ensemble pipeline using DataFrame-based evaluation approach.
This is optimized to pre-stack models and use efficient slicing.
"""

from zenml import pipeline
from steps.extract_ground_truth_labels import extract_ground_truth_labels
from steps.load_and_stack_all_predictions import load_and_stack_all_predictions
from steps.evaluate_ensemble_combinations_dataframe import evaluate_ensemble_combinations_dataframe


@pipeline(enable_cache=True)
def final_ensemble_pipeline_dataframe():
    """
    Optimized pipeline to evaluate all ensemble combinations.
    
    This pipeline:
    1. Extracts ground truth labels from test datasets
    2. Loads all 36 model predictions
    3. Pre-stacks models and evaluates all combinations efficiently
    """
    
    # Step 1: Extract ground truth labels
    temporal_labels, spatial_labels = extract_ground_truth_labels()
    
    # Step 2: Load and stack all predictions
    temporal_predictions, spatial_predictions = load_and_stack_all_predictions()
    
    # Step 3: Evaluate all combinations using DataFrame approach
    ensemble_results = evaluate_ensemble_combinations_dataframe(
        temporal_predictions=temporal_predictions,
        spatial_predictions=spatial_predictions,
        temporal_labels=temporal_labels,
        spatial_labels=spatial_labels,
        checkpoint_dir="results/ensemble_checkpoints",
        batch_size=100000
    )
    
    return ensemble_results


if __name__ == "__main__":
    # Run the pipeline
    pipeline_instance = final_ensemble_pipeline_dataframe()
    pipeline_run = pipeline_instance.run()
    
    print("Pipeline completed successfully!")
    print(f"Run name: {pipeline_run.name}")