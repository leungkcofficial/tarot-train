"""
Final ensemble pipeline that evaluates ALL possible combinations.

This pipeline:
1. Loads stacked DeepSurv predictions (12 groups with shape 2,5,n_samples)
2. Loads DeepHit predictions (12 models with shape 2,5,n_samples)
3. Evaluates ALL 16,777,191 possible ensemble combinations
4. Saves results with checkpointing for reliability
"""

from zenml import pipeline
from zenml.config import DockerSettings

from steps.extract_ground_truth_labels import extract_ground_truth_labels
from steps.load_and_stack_all_predictions import load_and_stack_all_predictions
from steps.evaluate_ensemble_combinations_parallel_efficient import evaluate_ensemble_combinations_parallel_efficient

docker_settings = DockerSettings(
    requirements=[
        "pandas",
        "numpy",
        "scikit-learn",
        "torch",
        "pycox",
        "h5py",
        "matplotlib",
        "seaborn",
        "lifelines",
        "psutil"
    ],
)


@pipeline(enable_cache=True, settings={"docker": docker_settings})
def final_ensemble_pipeline_full():
    """Final ensemble pipeline evaluating ALL combinations."""
    
    # Import data processing steps (same as ensemble_deploy_pipeline)
    from steps.ingest_data import ingest_data
    from steps.clean_data import clean_data
    from steps.merge_data import merge_data
    from steps.split_data import split_data
    from steps.impute_data import impute_data
    from steps.preprocess_data import preprocess_data
    
    # ==================== Data Processing Steps (Cached) ====================
    # These steps are identical to ensemble_deploy_pipeline to ensure consistency
    
    # Step 1: Ingest data from multiple sources
    cr_df, hb_df, a1c_df, alb_df, po4_df, ca_df, ca_adjusted_df, hco3_df, upcr_df, uacr_df, demo_df, icd10_df, death_df, operation_df = ingest_data()
    
    # Step 2: Clean all dataframes
    patient_df, icd10_df_clean, cr_df_clean, hb_df_clean, a1c_df_clean, alb_df_clean, po4_df_clean, ca_df_clean, ca_adjusted_df_clean, hco3_df_clean, upcr_df_clean, uacr_df_clean, operation_df_clean, death_df_clean, cci_df, cci_score_df, hypertension_df, egfr_df = clean_data(
        cr_df=cr_df,
        hb_df=hb_df,
        a1c_df=a1c_df,
        alb_df=alb_df,
        po4_df=po4_df,
        ca_df=ca_df,
        ca_adjusted_df=ca_adjusted_df,
        hco3_df=hco3_df,
        upcr_df=upcr_df,
        uacr_df=uacr_df,
        icd10_df=icd10_df,
        operation_df=operation_df,
        death_df=death_df,
        demo_df=demo_df
    )
    
    # Step 3: Merge cleaned dataframes
    final_df, prediction_df = merge_data(
        patient_df=patient_df,
        icd10_df=icd10_df_clean,
        cr_df=cr_df_clean,
        hb_df=hb_df_clean,
        a1c_df=a1c_df_clean,
        alb_df=alb_df_clean,
        po4_df=po4_df_clean,
        ca_df=ca_df_clean,
        ca_adjusted_df=ca_adjusted_df_clean,
        hco3_df=hco3_df_clean,
        upcr_df=upcr_df_clean,
        uacr_df=uacr_df_clean,
        operation_df=operation_df_clean,
        death_df=death_df_clean,
        cci_df=cci_df,
        cci_score_df=cci_score_df,
        hypertension_df=hypertension_df,
        egfr_df=egfr_df
    )
    
    # Step 4: Split data
    train_df, temporal_test_df, spatial_test_df, raw_df = split_data(
        raw_df=final_df,
        prediction_df=prediction_df
    )

    # Step 5: Impute missing data
    train_df_imputed, temporal_test_df_imputed, spatial_test_df_imputed = impute_data(
        train_df=train_df,
        temporal_test_df=temporal_test_df,
        spatial_test_df=spatial_test_df
    )
    
    # Step 6: Preprocess data
    train_df_preprocessed, temporal_test_df_preprocessed, spatial_test_df_preprocessed = preprocess_data(
        train_df=train_df_imputed,
        temporal_test_df=temporal_test_df_imputed,
        spatial_test_df=spatial_test_df_imputed
    )
    
    # ==================== Extract Ground Truth ====================
    # Extract ground truth labels from test datasets
    y_temporal_test, y_spatial_test = extract_ground_truth_labels(
        temporal_test_df=temporal_test_df,
        spatial_test_df=spatial_test_df
    )
    
    # ==================== Load and stack Predictions ====================
    # Load all 36 model predictions and stack them according to model grouping
    temporal_cif_all, spatial_cif_all, model_info = load_and_stack_all_predictions()
    
    # ==================== Evaluate ALL Ensemble Combinations ====================
    # Evaluate ALL possible ensemble combinations using parallel processing with efficient memory usage
    evaluation_results = evaluate_ensemble_combinations_parallel_efficient(
        temporal_cif_all=temporal_cif_all,
        spatial_cif_all=spatial_cif_all,
        y_temporal_test=y_temporal_test,
        y_spatial_test=y_spatial_test,
        model_info=model_info,
        batch_size=10000,  # Process in batches of 10k
        n_jobs=-1,  # Use all available CPU cores
        checkpoint_interval=50000  # Save checkpoint every 50k combinations
    )
    
    return evaluation_results