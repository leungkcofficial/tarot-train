# Metric Calculator Design Document

## Overview

The Metric Calculator is a standalone Python script that calculates evaluation metrics for CKD risk prediction models. It works with HDF5 prediction files and CSV/JSON SHAP value files to calculate metrics such as C-index, Brier score, and log likelihood, and to analyze SHAP values for feature importance.

## File Structure

The script will be located at `src/metric_calculator.py` and will have the following structure:

```
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Metric Calculator for CKD Risk Prediction Models

This script calculates evaluation metrics (C-index, Brier score, log likelihood)
and analyzes SHAP values from prediction files. It works with HDF5 prediction files
and CSV/JSON SHAP value files.

Usage:
    python metric_calculator.py --predictions <path_to_h5_file> [options]
    python metric_calculator.py --help

Options:
    --predictions PATH       Path to HDF5 file containing survival predictions
    --durations PATH         Path to CSV file containing event times (optional if included in metadata)
    --events PATH            Path to CSV file containing event indicators (optional if included in metadata)
    --shap PATH              Path to CSV or JSON file containing SHAP values (optional)
    --time-horizons DAYS     Comma-separated list of time horizons in days (default: 365,730,1095,1460,1825)
    --output-dir PATH        Directory to save output files (default: current directory)
    --n-bootstrap INT        Number of bootstrap iterations (default: 50)
    --visualize              Generate visualization plots (default: False)
    --verbose                Print detailed information (default: False)
"""

# Imports
# ...

# Functions
# ...

# Main function
# ...

# Entry point
if __name__ == "__main__":
    main()
```

## Functions

The script will include the following functions:

### 1. `parse_arguments()`

Parse command-line arguments using `argparse`.

```python
def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Calculate evaluation metrics for CKD risk prediction models.')
    
    # Required arguments
    parser.add_argument('--predictions', type=str, required=True,
                        help='Path to HDF5 file containing survival predictions')
    
    # Optional arguments
    parser.add_argument('--durations', type=str,
                        help='Path to CSV file containing event times')
    parser.add_argument('--events', type=str,
                        help='Path to CSV file containing event indicators')
    parser.add_argument('--shap', type=str,
                        help='Path to CSV or JSON file containing SHAP values')
    parser.add_argument('--time-horizons', type=str, default='365,730,1095,1460,1825',
                        help='Comma-separated list of time horizons in days (default: 365,730,1095,1460,1825)')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Directory to save output files (default: current directory)')
    parser.add_argument('--n-bootstrap', type=int, default=50,
                        help='Number of bootstrap iterations (default: 50)')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization plots (default: False)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed information (default: False)')
    
    return parser.parse_args()
```

### 2. `load_data(args)`

Load data from input files.

```python
def load_data(args):
    """
    Load data from input files.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Tuple of (predictions, durations, events, shap_values)
    """
    # Load predictions from HDF5 file
    # Load durations and events from CSV files or metadata
    # Load SHAP values from CSV or JSON file
    # Return loaded data
```

### 3. `calculate_metrics(surv, durations, events, time_horizons, n_bootstrap=50, verbose=False)`

Calculate survival metrics with bootstrap confidence intervals.

```python
def calculate_metrics(surv, durations, events, time_horizons, n_bootstrap=50, verbose=False):
    """
    Calculate survival metrics with bootstrap confidence intervals.
    
    Args:
        surv: Survival predictions DataFrame
        durations: Event times
        events: Event indicators
        time_horizons: List of time horizons
        n_bootstrap: Number of bootstrap iterations
        verbose: Whether to print detailed information
        
    Returns:
        Dictionary of metrics with confidence intervals
    """
    # Create EvalSurv object
    # Calculate concordance index
    # Perform bootstrap to estimate confidence intervals
    # Calculate metrics at each time horizon
    # Return results dictionary
```

### 4. `analyze_shap_values(shap_values, verbose=False)`

Analyze SHAP values to determine feature importance.

```python
def analyze_shap_values(shap_values, verbose=False):
    """
    Analyze SHAP values to determine feature importance.
    
    Args:
        shap_values: Dictionary of SHAP values
        verbose: Whether to print detailed information
        
    Returns:
        DataFrame of feature importance
    """
    # Create DataFrame with feature names and mean SHAP values
    # Sort by absolute mean SHAP value
    # Print top features
    # Return DataFrame
```

### 5. `create_visualizations(metrics, shap_df, predictions, durations, events, time_horizons, output_dir, verbose=False)`

Create visualizations for the evaluation metrics.

```python
def create_visualizations(metrics, shap_df, predictions, durations, events, time_horizons, output_dir, verbose=False):
    """
    Create visualizations for the evaluation metrics.
    
    Args:
        metrics: Dictionary of metrics
        shap_df: DataFrame of SHAP values
        predictions: Survival predictions DataFrame
        durations: Event times
        events: Event indicators
        time_horizons: List of time horizons
        output_dir: Directory to save visualizations
        verbose: Whether to print detailed information
        
    Returns:
        Dictionary of paths to saved visualizations
    """
    # Create metrics by time plot
    # Create SHAP values plot
    # Create calibration plot
    # Return dictionary of paths to saved visualizations
```

### 6. Visualization Functions

#### 6.1. `plot_metrics_by_time(metrics, time_horizons=None, output_path=None)`

Create plot of metrics by time horizon.

```python
def plot_metrics_by_time(metrics, time_horizons=None, output_path=None):
    """
    Create plot of metrics by time horizon.
    
    Args:
        metrics: Dictionary containing evaluation metrics
        time_horizons: List of time horizons (default: None, uses metrics_by_horizon keys)
        output_path: Path to save the plot (default: None, displays the plot)
        
    Returns:
        Path to saved plot if output_path is provided, None otherwise
    """
    # Extract metrics by horizon
    # Create figure
    # Plot Brier scores
    # Add integrated metrics as text
    # Save or display the plot
```

#### 6.2. `plot_shap_values(shap_df, top_n=10, output_path=None)`

Create visualization of SHAP values.

```python
def plot_shap_values(shap_df, top_n=10, output_path=None):
    """
    Create visualization of SHAP values.
    
    Args:
        shap_df: DataFrame containing SHAP values
        top_n: Number of top features to display (default: 10)
        output_path: Path to save the plot (default: None, displays the plot)
        
    Returns:
        Path to saved plot if output_path is provided, None otherwise
    """
    # Sort features by absolute SHAP value
    # Create figure
    # Create horizontal bar plot
    # Add error bars for confidence intervals
    # Save or display the plot
```

#### 6.3. `plot_calibration(predictions, durations, events, time_horizons=None, output_path=None)`

Create calibration plot for survival predictions.

```python
def plot_calibration(predictions, durations, events, time_horizons=None, output_path=None):
    """
    Create calibration plot for survival predictions.
    
    Args:
        predictions: DataFrame containing survival predictions
        durations: Array of event times
        events: Array of event indicators
        time_horizons: List of time horizons (default: [365, 730, 1095, 1460, 1825])
        output_path: Path to save the plot (default: None, displays the plot)
        
    Returns:
        Path to saved plot if output_path is provided, None otherwise
    """
    # Create figure with subplots
    # Plot each time horizon
    # Calculate observed risks using Kaplan-Meier estimator
    # Plot predicted vs. observed risks
    # Save or display the plot
```

### 7. `save_results(metrics, shap_df, visualization_paths, output_dir, verbose=False)`

Save results to files.

```python
def save_results(metrics, shap_df, visualization_paths, output_dir, verbose=False):
    """
    Save results to files.
    
    Args:
        metrics: Dictionary of metrics
        shap_df: DataFrame of SHAP values
        visualization_paths: Dictionary of paths to saved visualizations
        output_dir: Directory to save results
        verbose: Whether to print detailed information
        
    Returns:
        Dictionary of paths to saved files
    """
    # Save metrics to JSON
    # Save SHAP values to CSV
    # Create results dictionary
    # Save results paths to JSON
    # Return dictionary of paths to saved files
```

### 8. `main()`

Main function to tie everything together.

```python
def main():
    """Main function."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Load data
    predictions, durations, events, shap_values = load_data(args)
    
    # Parse time horizons
    time_horizons = [int(t) for t in args.time_horizons.split(',')]
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, durations, events, time_horizons, args.n_bootstrap, args.verbose)
    
    # Analyze SHAP values
    shap_df = analyze_shap_values(shap_values, args.verbose)
    
    # Create visualizations if requested
    visualization_paths = {}
    if args.visualize:
        visualization_paths = create_visualizations(metrics, shap_df, predictions, durations, events, time_horizons, args.output_dir, args.verbose)
    
    # Save results
    results_paths = save_results(metrics, shap_df, visualization_paths, args.output_dir, args.verbose)
    
    # Print summary
    if args.verbose:
        print("\nSummary:")
        print(f"C-index: {metrics['c_index']['mean']:.4f} (95% CI: {metrics['c_index']['lower']:.4f}-{metrics['c_index']['upper']:.4f})")
        print(f"Integrated Brier Score: {metrics['integrated_brier_score']['mean']:.4f} (95% CI: {metrics['integrated_brier_score']['lower']:.4f}-{metrics['integrated_brier_score']['upper']:.4f})")
        print(f"Integrated NBLL: {metrics['integrated_nbll']['mean']:.4f} (95% CI: {metrics['integrated_nbll']['lower']:.4f}-{metrics['integrated_nbll']['upper']:.4f})")
        
        if shap_df is not None:
            print("\nTop 5 features by SHAP value:")
            for i, row in shap_df.head(5).iterrows():
                ci_str = f" (95% CI: {row['lower_ci']:.4f} to {row['upper_ci']:.4f})" if 'lower_ci' in shap_df.columns else ""
                print(f"{row['feature']}: {row['mean_shap']:.4f}{ci_str}")
    
    print(f"\nResults saved to: {args.output_dir}")
```

## Error Handling and Validation

The script will include comprehensive error handling and validation for input files to ensure robustness and provide clear error messages to users. The following error handling and validation mechanisms will be implemented:

### 1. Input File Validation

#### 1.1. Predictions File (HDF5)

```python
def validate_predictions_file(file_path):
    """
    Validate the predictions HDF5 file.
    
    Args:
        file_path: Path to the HDF5 file
        
    Returns:
        True if valid, raises an exception otherwise
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Predictions file not found: {file_path}")
    
    # Check if file is a valid HDF5 file
    try:
        with h5py.File(file_path, 'r') as f:
            # Check if required datasets exist
            if 'predictions' not in f:
                raise ValueError(f"Missing 'predictions' dataset in HDF5 file: {file_path}")
            
            # Check if predictions dataset has the correct shape
            predictions = f['predictions']
            if len(predictions.shape) != 2:
                raise ValueError(f"Predictions dataset must be 2-dimensional, got shape {predictions.shape}")
            
            # Check if index and columns datasets exist
            if 'index' not in f:
                raise ValueError(f"Missing 'index' dataset in HDF5 file: {file_path}")
            
            if 'columns' not in f:
                raise ValueError(f"Missing 'columns' dataset in HDF5 file: {file_path}")
            
            # Check if index and columns have the correct shape
            index = f['index']
            columns = f['columns']
            
            if len(index) != predictions.shape[0]:
                raise ValueError(f"Index length ({len(index)}) does not match predictions shape ({predictions.shape})")
            
            if len(columns) != predictions.shape[1]:
                raise ValueError(f"Columns length ({len(columns)}) does not match predictions shape ({predictions.shape})")
    except Exception as e:
        if isinstance(e, (FileNotFoundError, ValueError)):
            raise
        else:
            raise ValueError(f"Invalid HDF5 file: {file_path}. Error: {str(e)}")
    
    return True
```

#### 1.2. Durations and Events Files (CSV)

```python
def validate_durations_events_files(durations_path, events_path):
    """
    Validate the durations and events CSV files.
    
    Args:
        durations_path: Path to the durations CSV file
        events_path: Path to the events CSV file
        
    Returns:
        True if valid, raises an exception otherwise
    """
    # Check if files exist
    if not os.path.exists(durations_path):
        raise FileNotFoundError(f"Durations file not found: {durations_path}")
    
    if not os.path.exists(events_path):
        raise FileNotFoundError(f"Events file not found: {events_path}")
    
    # Check if files are valid CSV files
    try:
        durations_df = pd.read_csv(durations_path)
        events_df = pd.read_csv(events_path)
        
        # Check if files have at least one column
        if durations_df.shape[1] == 0:
            raise ValueError(f"Durations file has no columns: {durations_path}")
        
        if events_df.shape[1] == 0:
            raise ValueError(f"Events file has no columns: {events_path}")
        
        # Check if files have the same number of rows
        if durations_df.shape[0] != events_df.shape[0]:
            raise ValueError(f"Durations file ({durations_df.shape[0]} rows) and events file ({events_df.shape[0]} rows) have different numbers of rows")
        
        # Check if durations are numeric
        if not pd.to_numeric(durations_df.iloc[:, 0], errors='coerce').notna().all():
            raise ValueError(f"Durations file contains non-numeric values: {durations_path}")
        
        # Check if events are binary (0 or 1)
        events = events_df.iloc[:, 0]
        if not ((events == 0) | (events == 1)).all():
            raise ValueError(f"Events file contains non-binary values (must be 0 or 1): {events_path}")
    except Exception as e:
        if isinstance(e, (FileNotFoundError, ValueError)):
            raise
        else:
            raise ValueError(f"Invalid CSV file(s). Error: {str(e)}")
    
    return True
```

#### 1.3. SHAP Values File (CSV or JSON)

```python
def validate_shap_file(file_path):
    """
    Validate the SHAP values file (CSV or JSON).
    
    Args:
        file_path: Path to the SHAP values file
        
    Returns:
        True if valid, raises an exception otherwise
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"SHAP values file not found: {file_path}")
    
    # Check file extension
    if file_path.endswith('.json'):
        # Validate JSON file
        try:
            with open(file_path, 'r') as f:
                shap_values = json.load(f)
            
            # Check if required keys exist
            required_keys = ['feature_names', 'mean_shap']
            for key in required_keys:
                if key not in shap_values:
                    raise ValueError(f"Missing required key '{key}' in SHAP values JSON file: {file_path}")
            
            # Check if feature_names and mean_shap have the same length
            if len(shap_values['feature_names']) != len(shap_values['mean_shap']):
                raise ValueError(f"Length mismatch between feature_names ({len(shap_values['feature_names'])}) and mean_shap ({len(shap_values['mean_shap'])}) in SHAP values JSON file: {file_path}")
        except Exception as e:
            if isinstance(e, (FileNotFoundError, ValueError)):
                raise
            else:
                raise ValueError(f"Invalid JSON file: {file_path}. Error: {str(e)}")
    elif file_path.endswith('.csv'):
        # Validate CSV file
        try:
            shap_df = pd.read_csv(file_path)
            
            # Check if required columns exist
            required_columns = ['feature', 'mean_shap']
            for column in required_columns:
                if column not in shap_df.columns:
                    raise ValueError(f"Missing required column '{column}' in SHAP values CSV file: {file_path}")
        except Exception as e:
            if isinstance(e, (FileNotFoundError, ValueError)):
                raise
            else:
                raise ValueError(f"Invalid CSV file: {file_path}. Error: {str(e)}")
    else:
        raise ValueError(f"Unsupported file format for SHAP values: {file_path}. Must be .json or .csv")
    
    return True
```

### 2. Data Consistency Validation

```python
def validate_data_consistency(predictions, durations, events):
    """
    Validate the consistency between predictions, durations, and events.
    
    Args:
        predictions: Survival predictions DataFrame
        durations: Event times
        events: Event indicators
        
    Returns:
        True if valid, raises an exception otherwise
    """
    # Check if durations and events have the same length
    if len(durations) != len(events):
        raise ValueError(f"Length mismatch between durations ({len(durations)}) and events ({len(events)})")
    
    # Check if predictions has the correct shape
    if len(durations) != predictions.shape[1] and len(durations) != predictions.shape[0]:
        raise ValueError(f"Dimension mismatch between durations ({len(durations)}) and predictions ({predictions.shape})")
    
    # Check if durations are non-negative
    if (durations < 0).any():
        raise ValueError("Durations must be non-negative")
    
    # Check if events are binary (0 or 1)
    if not ((events == 0) | (events == 1)).all():
        raise ValueError("Events must be binary (0 or 1)")
    
    # Check if predictions are probabilities (between 0 and 1)
    if (predictions < 0).any().any() or (predictions > 1).any().any():
        raise ValueError("Predictions must be probabilities (between 0 and 1)")
    
    return True
```

### 3. Time Horizons Validation

```python
def validate_time_horizons(time_horizons, durations):
    """
    Validate the time horizons.
    
    Args:
        time_horizons: List of time horizons
        durations: Event times
        
    Returns:
        True if valid, raises an exception otherwise
    """
    # Check if time horizons are positive
    if any(t <= 0 for t in time_horizons):
        raise ValueError("Time horizons must be positive")
    
    # Check if time horizons are in ascending order
    if not all(time_horizons[i] < time_horizons[i+1] for i in range(len(time_horizons)-1)):
        raise ValueError("Time horizons must be in ascending order")
    
    # Check if time horizons are within the range of durations
    if max(time_horizons) > max(durations) * 1.5:
        print(f"Warning: Maximum time horizon ({max(time_horizons)}) is greater than 1.5 times the maximum duration ({max(durations)})")
    
    return True
```

### 4. Output Directory Validation

```python
def validate_output_directory(output_dir):
    """
    Validate the output directory.
    
    Args:
        output_dir: Path to the output directory
        
    Returns:
        True if valid, raises an exception otherwise
    """
    # Check if output directory exists, create it if it doesn't
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            raise ValueError(f"Failed to create output directory: {output_dir}. Error: {str(e)}")
    
    # Check if output directory is writable
    if not os.access(output_dir, os.W_OK):
        raise ValueError(f"Output directory is not writable: {output_dir}")
    
    return True
```

### 5. Integration in Main Function

The validation functions will be integrated into the main function to ensure that all inputs are valid before proceeding with the analysis:

```python
def main():
    """Main function."""
    try:
        # Parse command-line arguments
        args = parse_arguments()
        
        # Validate output directory
        validate_output_directory(args.output_dir)
        
        # Load and validate data
        predictions, durations, events, shap_values = load_data(args)
        
        # Validate data consistency
        validate_data_consistency(predictions, durations, events)
        
        # Parse and validate time horizons
        time_horizons = [int(t) for t in args.time_horizons.split(',')]
        validate_time_horizons(time_horizons, durations)
        
        # Calculate metrics
        metrics = calculate_metrics(predictions, durations, events, time_horizons, args.n_bootstrap, args.verbose)
        
        # Analyze SHAP values
        shap_df = analyze_shap_values(shap_values, args.verbose)
        
        # Create visualizations if requested
        visualization_paths = {}
        if args.visualize:
            visualization_paths = create_visualizations(metrics, shap_df, predictions, durations, events, time_horizons, args.output_dir, args.verbose)
        
        # Save results
        results_paths = save_results(metrics, shap_df, visualization_paths, args.output_dir, args.verbose)
        
        # Print summary
        if args.verbose:
            print("\nSummary:")
            print(f"C-index: {metrics['c_index']['mean']:.4f} (95% CI: {metrics['c_index']['lower']:.4f}-{metrics['c_index']['upper']:.4f})")
            print(f"Integrated Brier Score: {metrics['integrated_brier_score']['mean']:.4f} (95% CI: {metrics['integrated_brier_score']['lower']:.4f}-{metrics['integrated_brier_score']['upper']:.4f})")
            print(f"Integrated NBLL: {metrics['integrated_nbll']['mean']:.4f} (95% CI: {metrics['integrated_nbll']['lower']:.4f}-{metrics['integrated_nbll']['upper']:.4f})")
            
            if shap_df is not None:
                print("\nTop 5 features by SHAP value:")
                for i, row in shap_df.head(5).iterrows():
                    ci_str = f" (95% CI: {row['lower_ci']:.4f} to {row['upper_ci']:.4f})" if 'lower_ci' in shap_df.columns else ""
                    print(f"{row['feature']}: {row['mean_shap']:.4f}{ci_str}")
        
        print(f"\nResults saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
```

## Usage Guide

### Installation

Before using the Metric Calculator, ensure you have the required dependencies installed:

```bash
pip install numpy pandas matplotlib pycox lifelines h5py
```

### Basic Usage

The Metric Calculator can be used with a single command to calculate evaluation metrics for a CKD risk prediction model:

```bash
python src/metric_calculator.py --predictions results/model_evaluation/temporal_test_predictions_20250711_063843.h5 --visualize
```

This command will:
1. Load the predictions from the specified HDF5 file
2. Extract durations and events from the metadata in the HDF5 file
3. Calculate evaluation metrics (C-index, Brier score, log likelihood)
4. Generate visualization plots
5. Save the results to the current directory

### Command-Line Options

The Metric Calculator supports the following command-line options:

#### Required Arguments

- `--predictions PATH`: Path to the HDF5 file containing survival predictions. This file should be created using the `save_predictions_to_hdf5` function from `src/util.py`.

#### Optional Arguments

- `--durations PATH`: Path to the CSV file containing event times. If not provided, the script will attempt to extract durations from the metadata in the HDF5 file.
- `--events PATH`: Path to the CSV file containing event indicators. If not provided, the script will attempt to extract events from the metadata in the HDF5 file.
- `--shap PATH`: Path to the CSV or JSON file containing SHAP values. If provided, the script will analyze SHAP values to determine feature importance.
- `--time-horizons DAYS`: Comma-separated list of time horizons in days. Default: `365,730,1095,1460,1825` (1-5 years).
- `--output-dir PATH`: Directory to save output files. Default: current directory.
- `--n-bootstrap INT`: Number of bootstrap iterations for confidence interval estimation. Default: `50`.
- `--visualize`: Generate visualization plots. Default: `False`.
- `--verbose`: Print detailed information during execution. Default: `False`.

### Examples

#### Basic Usage

```bash
python src/metric_calculator.py --predictions results/model_evaluation/temporal_test_predictions_20250711_063843.h5 --visualize
```

#### Specifying Durations and Events

If durations and events are not included in the metadata of the HDF5 file, you can specify them explicitly:

```bash
python src/metric_calculator.py --predictions results/model_evaluation/temporal_test_predictions_20250711_063843.h5 --durations data/durations.csv --events data/events.csv --visualize
```

#### Including SHAP Values

To analyze SHAP values for feature importance:

```bash
python src/metric_calculator.py --predictions results/model_evaluation/temporal_test_predictions_20250711_063843.h5 --shap results/model_evaluation/temporal_test_shap_values_20250711_064118.json --visualize
```

#### Custom Time Horizons

To specify custom time horizons:

```bash
python src/metric_calculator.py --predictions results/model_evaluation/temporal_test_predictions_20250711_063843.h5 --time-horizons 365,730,1095 --visualize
```

#### Verbose Output

To print detailed information during execution:

```bash
python src/metric_calculator.py --predictions results/model_evaluation/temporal_test_predictions_20250711_063843.h5 --visualize --verbose
```

#### Custom Output Directory

To save output files to a specific directory:

```bash
python src/metric_calculator.py --predictions results/model_evaluation/temporal_test_predictions_20250711_063843.h5 --output-dir results/metrics --visualize
```

### Output Files

The Metric Calculator generates the following output files:

1. `metrics_TIMESTAMP.json`: JSON file containing evaluation metrics (C-index, Brier score, log likelihood) with confidence intervals.
2. `shap_values_TIMESTAMP.csv`: CSV file containing SHAP values for feature importance (if SHAP values are provided).
3. `metrics_by_time_TIMESTAMP.png`: Plot of Brier scores by time horizon (if `--visualize` is specified).
4. `shap_values_TIMESTAMP.png`: Plot of SHAP values for feature importance (if SHAP values are provided and `--visualize` is specified).
5. `calibration_TIMESTAMP.png`: Calibration plot showing predicted vs. observed risks at different time horizons (if `--visualize` is specified).

### Interpreting Results

#### Evaluation Metrics

- **C-index**: Concordance index, a measure of the model's discriminative ability. Higher values indicate better discrimination, with 1.0 being perfect discrimination and 0.5 being no better than random.
- **Brier score**: A measure of the model's calibration. Lower values indicate better calibration, with 0.0 being perfect calibration.
- **Log likelihood**: A measure of the model's overall fit. Higher values indicate better fit.

#### SHAP Values

SHAP (SHapley Additive exPlanations) values indicate the contribution of each feature to the model's predictions. Features with higher absolute SHAP values have a greater impact on the model's predictions.

## Future Enhancements

Potential future enhancements include:

1. Support for additional metrics (e.g., AUC, sensitivity, specificity)
2. Integration with other prediction file formats (e.g., CSV, JSON, pickle)
3. Interactive visualizations using Plotly or Bokeh
4. Batch processing of multiple prediction files
5. Comparison of multiple models
6. Integration with MLflow for experiment tracking
7. Web interface for easier use
8. Support for additional survival models (e.g., Cox-Time, Random Survival Forests)
9. Parallel processing for faster bootstrap calculations
10. Export of results to other formats (e.g., Excel, PDF)

## Troubleshooting

### Common Issues

1. **File not found**: Ensure that the paths to the input files are correct and that the files exist.
2. **Invalid file format**: Ensure that the input files have the correct format (HDF5 for predictions, CSV or JSON for SHAP values).
3. **Dimension mismatch**: Ensure that the dimensions of the predictions, durations, and events are consistent.
4. **Memory error**: If you encounter memory errors, try reducing the number of bootstrap iterations (`--n-bootstrap`) or processing the data in batches.
5. **Visualization error**: If you encounter errors when generating visualizations, ensure that you have the required dependencies installed (matplotlib).
6. **Time horizon error**: If you encounter errors related to time horizons, ensure that the time horizons are positive, in ascending order, and within a reasonable range of the durations.
7. **Bootstrap error**: If bootstrap calculations are taking too long, try reducing the number of bootstrap iterations or using a more powerful machine.

### Error Messages

The Metric Calculator provides detailed error messages to help diagnose issues. If you encounter an error, check the error message for information about the cause of the error and how to fix it.

### Getting Help

If you need help using the Metric Calculator, you can:

1. Run `python src/metric_calculator.py --help` to see the available command-line options.
2. Check the documentation in this design document for detailed information about the script's functionality.
3. Examine the source code for more detailed information about the implementation.
4. Contact the development team for assistance with complex issues.

## Conclusion

The Metric Calculator is a powerful tool for evaluating CKD risk prediction models. It provides a comprehensive set of evaluation metrics, including the C-index, Brier score, and log likelihood, as well as tools for analyzing SHAP values to determine feature importance. The script is designed to be easy to use, with a simple command-line interface and clear error messages. It also includes robust error handling and validation to ensure that the results are accurate and reliable.