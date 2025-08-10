"""
Script to run the generate predictions with baseline hazards pipeline.
"""

from pipelines.generate_predictions_with_baseline import generate_predictions_pipeline

if __name__ == "__main__":
    print("Starting Generate Predictions with Baseline Hazards Pipeline...")
    print("=" * 80)
    
    # Create and run the pipeline
    pipeline = generate_predictions_pipeline()
    pipeline.run()
    
    print("\n" + "=" * 80)
    print("Pipeline execution completed!")
    print("Check results/final_deploy/individual_predictions/ for the generated predictions.")