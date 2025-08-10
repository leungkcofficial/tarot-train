"""
Runner script for the final ensemble pipeline.
"""

from pipelines.final_ensemble_pipeline import final_ensemble_pipeline
import json
from datetime import datetime
import os


def main():
    """Run the final ensemble pipeline."""
    print("Starting final ensemble pipeline...")
    print("="*60)
    
    # Run the pipeline
    pipeline_run = final_ensemble_pipeline()
    
    # Get the evaluation results
    evaluation_results = pipeline_run.steps["evaluate_final_ensemble"].output.load()
    
    # Save results
    output_dir = "results/final_deploy"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"ensemble_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    with open(output_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("FINAL ENSEMBLE RESULTS")
    print("="*60)
    
    summary = evaluation_results.get("summary", {})
    print(f"Temporal Average C-index: {summary.get('temporal_avg_c_index', 'N/A'):.4f}")
    print(f"Spatial Average C-index: {summary.get('spatial_avg_c_index', 'N/A'):.4f}")
    print(f"Temporal Average IBS: {summary.get('temporal_avg_ibs', 'N/A'):.4f}")
    print(f"Spatial Average IBS: {summary.get('spatial_avg_ibs', 'N/A'):.4f}")
    
    print("\nDetailed Results:")
    
    # Temporal results
    print("\nTemporal Dataset:")
    for event, metrics in evaluation_results.get("temporal", {}).items():
        print(f"  {event}:")
        print(f"    C-index: {metrics.get('c_index', 'N/A'):.4f}")
        print(f"    IBS: {metrics.get('ibs', 'N/A'):.4f}")
    
    # Spatial results
    print("\nSpatial Dataset:")
    for event, metrics in evaluation_results.get("spatial", {}).items():
        print(f"  {event}:")
        print(f"    C-index: {metrics.get('c_index', 'N/A'):.4f}")
        print(f"    IBS: {metrics.get('ibs', 'N/A'):.4f}")
    
    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()