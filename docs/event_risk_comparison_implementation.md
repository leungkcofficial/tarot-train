# Event Risk Comparison Implementation Plan

## Overview

This document outlines the implementation plan for the `compare_event_risks` function to be added to `EDA.py`. The function will compare the risk of dialysis (event 1) and all-cause mortality (event 2) between the training, spatial test, and temporal test datasets.

## Function Purpose

The function will:
1. Classify patients based on UACR (A1, A2, A3) and eGFR (G3a, G3b, G4, G5) values
2. Calculate cumulative incidence of events using AalenJohansenFitter
3. Generate plots comparing risks across datasets (both discrete time points and continuous curves)
4. Create summary tables with 5-year risk percentages in both CSV and HTML formats

## Function Signature

```python
def compare_event_risks(
    train_df: pd.DataFrame,
    temporal_test_df: pd.DataFrame,
    spatial_test_df: pd.DataFrame,
    output_path: str
) -> Dict[str, Any]:
    """
    Compare the risk of dialysis (event 1) and all-cause mortality (event 2) between datasets.
    
    This function:
    1. Classifies patients based on UACR (A1, A2, A3) and eGFR (G3a, G3b, G4, G5) values
    2. Calculates cumulative incidence of events using AalenJohansenFitter
    3. Generates plots comparing risks across datasets (both discrete time points and continuous curves)
    4. Creates summary tables with 5-year risk percentages in both CSV and HTML formats
    
    Args:
        train_df: DataFrame containing the training data
        temporal_test_df: DataFrame containing the temporal test data
        spatial_test_df: DataFrame containing the spatial test data
        output_path: Path to save the visualizations and tables
        
    Returns:
        Dictionary containing paths to generated visualizations and tables
    """
```

## Implementation Steps

### 1. Data Preparation

- Make copies of input dataframes to avoid modifying originals
- Create output directory if it doesn't exist
- Initialize dictionaries to store risk percentages for summary tables
- Define functions to classify UACR and eGFR values
- Apply classifications to all datasets

```python
# Make copies of dataframes to avoid modifying originals
train_copy = train_df.copy()
spatial_test_copy = spatial_test_df.copy()
temporal_test_copy = temporal_test_df.copy()

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Initialize list to store visualization paths
visualization_paths = []

# Initialize dictionaries to store risk percentages for summary tables
dialysis_risk_summary = {
    'UACR Stage': [],
    'eGFR Stage': [],
    'Training Set (n)': [],
    'Training Set (%)': [],
    'Spatial Test Set (n)': [],
    'Spatial Test Set (%)': [],
    'Temporal Test Set (n)': [],
    'Temporal Test Set (%)': []
}

mortality_risk_summary = {
    'UACR Stage': [],
    'eGFR Stage': [],
    'Training Set (n)': [],
    'Training Set (%)': [],
    'Spatial Test Set (n)': [],
    'Spatial Test Set (%)': [],
    'Temporal Test Set (n)': [],
    'Temporal Test Set (%)': []
}

# Function to classify UACR values
def classify_uacr(df):
    df['uacr_stage'] = pd.NA
    df.loc[df['uacr'] < 30, 'uacr_stage'] = 'A1'
    df.loc[(df['uacr'] >= 30) & (df['uacr'] <= 300), 'uacr_stage'] = 'A2'
    df.loc[df['uacr'] > 300, 'uacr_stage'] = 'A3'
    return df

# Function to classify eGFR values
def classify_egfr(df):
    df['egfr_stage'] = pd.NA
    df.loc[(df['egfr'] >= 45) & (df['egfr'] < 60), 'egfr_stage'] = 'G3a'
    df.loc[(df['egfr'] >= 30) & (df['egfr'] < 45), 'egfr_stage'] = 'G3b'
    df.loc[(df['egfr'] >= 15) & (df['egfr'] < 30), 'egfr_stage'] = 'G4'
    df.loc[df['egfr'] < 15, 'egfr_stage'] = 'G5'
    return df

# Apply classifications to all datasets
for df in [train_copy, spatial_test_copy, temporal_test_copy]:
    classify_uacr(df)
    classify_egfr(df)
```

### 2. Create Plot Figures

- Define time points for evaluation (1-5 years)
- Create figures for discrete time points and continuous curves
- Set up row and column titles for all plots

```python
# Define time points for evaluation (in days)
time_points = [365, 730, 1095, 1460, 1825]  # 1-5 years
years = [1, 2, 3, 4, 5]  # For x-axis labels

# Define UACR and eGFR stages
uacr_stages = ['A1', 'A2', 'A3']
egfr_stages = ['G3a', 'G3b', 'G4', 'G5']

# Create figures for discrete time points
# Dialysis risk (event 1)
fig_dialysis_discrete, axes_dialysis_discrete = plt.subplots(3, 4, figsize=(20, 15), sharex=True, sharey=True)
fig_dialysis_discrete.suptitle('Risk of Dialysis (Event 1) by UACR and eGFR Stage - Yearly Time Points', fontsize=16)

# Mortality risk (event 2)
fig_mortality_discrete, axes_mortality_discrete = plt.subplots(3, 4, figsize=(20, 15), sharex=True, sharey=True)
fig_mortality_discrete.suptitle('Risk of All-Cause Mortality (Event 2) by UACR and eGFR Stage - Yearly Time Points', fontsize=16)

# Create figures for continuous curves
# Dialysis risk (event 1)
fig_dialysis_continuous, axes_dialysis_continuous = plt.subplots(3, 4, figsize=(20, 15), sharex=True, sharey=True)
fig_dialysis_continuous.suptitle('Risk of Dialysis (Event 1) by UACR and eGFR Stage - Continuous Curves', fontsize=16)

# Mortality risk (event 2)
fig_mortality_continuous, axes_mortality_continuous = plt.subplots(3, 4, figsize=(20, 15), sharex=True, sharey=True)
fig_mortality_continuous.suptitle('Risk of All-Cause Mortality (Event 2) by UACR and eGFR Stage - Continuous Curves', fontsize=16)

# Set up row and column titles for all plots
for i, uacr_stage in enumerate(uacr_stages):
    # Discrete plots
    axes_dialysis_discrete[i, 0].set_ylabel(f'UACR {uacr_stage}\nCumulative Incidence (%)', fontsize=12)
    axes_mortality_discrete[i, 0].set_ylabel(f'UACR {uacr_stage}\nCumulative Incidence (%)', fontsize=12)
    
    # Continuous plots
    axes_dialysis_continuous[i, 0].set_ylabel(f'UACR {uacr_stage}\nCumulative Incidence (%)', fontsize=12)
    axes_mortality_continuous[i, 0].set_ylabel(f'UACR {uacr_stage}\nCumulative Incidence (%)', fontsize=12)
    
for j, egfr_stage in enumerate(egfr_stages):
    # Discrete plots
    axes_dialysis_discrete[2, j].set_xlabel(f'Follow-up (years)\neGFR {egfr_stage}', fontsize=12)
    axes_mortality_discrete[2, j].set_xlabel(f'Follow-up (years)\neGFR {egfr_stage}', fontsize=12)
    
    # Continuous plots
    axes_dialysis_continuous[2, j].set_xlabel(f'Follow-up (days)\neGFR {egfr_stage}', fontsize=12)
    axes_mortality_continuous[2, j].set_xlabel(f'Follow-up (days)\neGFR {egfr_stage}', fontsize=12)
```

### 3. Define Helper Functions

- Function to calculate cumulative incidence using AalenJohansenFitter
- Functions to plot discrete time points and continuous curves

```python
# Function to calculate cumulative incidence using AalenJohansenFitter
def calculate_cumulative_incidence(df, event_of_interest, time_points):
    if len(df) == 0:
        return [np.nan] * len(time_points), None
        
    durations = df['time'].values
    events = df['endpoint'].values
    
    try:
        ajf = AalenJohansenFitter()
        ajf.fit(durations, events, event_of_interest=event_of_interest)
        
        # Calculate risk at specific time points (1-5 years)
        risk_values = []
        
        for days in time_points:
            try:
                # Get the risk at this time point (as percentage)
                risk = ajf.cumulative_density_at_times(days)[0] * 100
                risk_values.append(risk)
            except:
                risk_values.append(np.nan)
                
        return risk_values, ajf
    except Exception as e:
        print(f"Error calculating cumulative incidence: {e}")
        return [np.nan] * len(time_points), None

# Function to plot discrete time points
def plot_discrete_points(ax, years, risk_values, dataset_name, color):
    if all(np.isnan(risk_values)):
        return
        
    ax.plot(years, risk_values, marker='o', linestyle='-', label=dataset_name, color=color)

# Function to plot continuous curve
def plot_continuous_curve(ax, ajf, dataset_name, color):
    if ajf is None:
        return
        
    # Plot the cumulative incidence curve (as percentage)
    ajf.plot(ax=ax, color=color, label=dataset_name, ci_show=False)
    
    # Convert y-axis to percentage
    y_ticks = ax.get_yticks()
    ax.set_yticklabels([f"{tick*100:.0f}" for tick in y_ticks])
```

### 4. Calculate and Plot Risks

- Loop through each combination of UACR and eGFR stage
- Calculate dialysis and mortality risks
- Plot discrete time points and continuous curves
- Add sample size annotations
- Add to summary tables

```python
# Initialize data for HTML tables
dialysis_html_data = []
mortality_html_data = []

# Loop through each combination of UACR and eGFR stage
for i, uacr_stage in enumerate(uacr_stages):
    for j, egfr_stage in enumerate(egfr_stages):
        # Get subsets for each dataset with the current UACR and eGFR stage
        train_subset = train_copy[(train_copy['uacr_stage'] == uacr_stage) & 
                                 (train_copy['egfr_stage'] == egfr_stage)]
        
        spatial_subset = spatial_test_copy[(spatial_test_copy['uacr_stage'] == uacr_stage) & 
                                          (spatial_test_copy['egfr_stage'] == egfr_stage)]
        
        temporal_subset = temporal_test_copy[(temporal_test_copy['uacr_stage'] == uacr_stage) & 
                                            (temporal_test_copy['egfr_stage'] == egfr_stage)]
        
        # Calculate dialysis risk (event 1)
        train_dialysis_risk, train_dialysis_ajf = calculate_cumulative_incidence(train_subset, 1, time_points)
        spatial_dialysis_risk, spatial_dialysis_ajf = calculate_cumulative_incidence(spatial_subset, 1, time_points)
        temporal_dialysis_risk, temporal_dialysis_ajf = calculate_cumulative_incidence(temporal_subset, 1, time_points)
        
        # Calculate mortality risk (event 2)
        train_mortality_risk, train_mortality_ajf = calculate_cumulative_incidence(train_subset, 2, time_points)
        spatial_mortality_risk, spatial_mortality_ajf = calculate_cumulative_incidence(spatial_subset, 2, time_points)
        temporal_mortality_risk, temporal_mortality_ajf = calculate_cumulative_incidence(temporal_subset, 2, time_points)
        
        # Plot discrete time points for dialysis risk
        plot_discrete_points(axes_dialysis_discrete[i, j], years, train_dialysis_risk, 'Training Set', 'blue')
        plot_discrete_points(axes_dialysis_discrete[i, j], years, spatial_dialysis_risk, 'Spatial Test Set', 'green')
        plot_discrete_points(axes_dialysis_discrete[i, j], years, temporal_dialysis_risk, 'Temporal Test Set', 'orange')
        
        # Plot discrete time points for mortality risk
        plot_discrete_points(axes_mortality_discrete[i, j], years, train_mortality_risk, 'Training Set', 'blue')
        plot_discrete_points(axes_mortality_discrete[i, j], years, spatial_mortality_risk, 'Spatial Test Set', 'green')
        plot_discrete_points(axes_mortality_discrete[i, j], years, temporal_mortality_risk, 'Temporal Test Set', 'orange')
        
        # Plot continuous curves for dialysis risk
        plot_continuous_curve(axes_dialysis_continuous[i, j], train_dialysis_ajf, 'Training Set', 'blue')
        plot_continuous_curve(axes_dialysis_continuous[i, j], spatial_dialysis_ajf, 'Spatial Test Set', 'green')
        plot_continuous_curve(axes_dialysis_continuous[i, j], temporal_dialysis_ajf, 'Temporal Test Set', 'orange')
        
        # Plot continuous curves for mortality risk
        plot_continuous_curve(axes_mortality_continuous[i, j], train_mortality_ajf, 'Training Set', 'blue')
        plot_continuous_curve(axes_mortality_continuous[i, j], spatial_mortality_ajf, 'Spatial Test Set', 'green')
        plot_continuous_curve(axes_mortality_continuous[i, j], temporal_mortality_ajf, 'Temporal Test Set', 'orange')
        
        # Add grid and set y-axis limit for all plots
        # Discrete plots
        axes_dialysis_discrete[i, j].grid(True, alpha=0.3)
        axes_mortality_discrete[i, j].grid(True, alpha=0.3)
        axes_dialysis_discrete[i, j].set_ylim(0, 100)
        axes_mortality_discrete[i, j].set_ylim(0, 100)
        
        # Continuous plots
        axes_dialysis_continuous[i, j].grid(True, alpha=0.3)
        axes_mortality_continuous[i, j].grid(True, alpha=0.3)
        axes_dialysis_continuous[i, j].set_ylim(0, 1)  # 0-100%
        axes_mortality_continuous[i, j].set_ylim(0, 1)  # 0-100%
        
        # Add sample size annotation to all plots
        sample_text = f"n={len(train_subset)}/{len(spatial_subset)}/{len(temporal_subset)}"
        
        # Discrete plots
        axes_dialysis_discrete[i, j].annotate(sample_text, xy=(0.5, 0.95), xycoords='axes fraction', 
                                            ha='center', fontsize=8)
        axes_mortality_discrete[i, j].annotate(sample_text, xy=(0.5, 0.95), xycoords='axes fraction', 
                                             ha='center', fontsize=8)
        
        # Continuous plots
        axes_dialysis_continuous[i, j].annotate(sample_text, xy=(0.5, 0.95), xycoords='axes fraction', 
                                              ha='center', fontsize=8)
        axes_mortality_continuous[i, j].annotate(sample_text, xy=(0.5, 0.95), xycoords='axes fraction', 
                                               ha='center', fontsize=8)
        
        # Format 5-year risk values for display
        train_dialysis_5yr = train_dialysis_risk[-1] if not np.isnan(train_dialysis_risk[-1]) else np.nan
        spatial_dialysis_5yr = spatial_dialysis_risk[-1] if not np.isnan(spatial_dialysis_risk[-1]) else np.nan
        temporal_dialysis_5yr = temporal_dialysis_risk[-1] if not np.isnan(temporal_dialysis_risk[-1]) else np.nan
        
        train_mortality_5yr = train_mortality_risk[-1] if not np.isnan(train_mortality_risk[-1]) else np.nan
        spatial_mortality_5yr = spatial_mortality_risk[-1] if not np.isnan(spatial_mortality_risk[-1]) else np.nan
        temporal_mortality_5yr = temporal_mortality_risk[-1] if not np.isnan(temporal_mortality_risk[-1]) else np.nan
        
        # Add to CSV summary tables (5-year risk)
        dialysis_risk_summary['UACR Stage'].append(uacr_stage)
        dialysis_risk_summary['eGFR Stage'].append(egfr_stage)
        dialysis_risk_summary['Training Set (n)'].append(len(train_subset))
        dialysis_risk_summary['Training Set (%)'].append(train_dialysis_5yr)
        dialysis_risk_summary['Spatial Test Set (n)'].append(len(spatial_subset))
        dialysis_risk_summary['Spatial Test Set (%)'].append(spatial_dialysis_5yr)
        dialysis_risk_summary['Temporal Test Set (n)'].append(len(temporal_subset))
        dialysis_risk_summary['Temporal Test Set (%)'].append(temporal_dialysis_5yr)
        
        mortality_risk_summary['UACR Stage'].append(uacr_stage)
        mortality_risk_summary['eGFR Stage'].append(egfr_stage)
        mortality_risk_summary['Training Set (n)'].append(len(train_subset))
        mortality_risk_summary['Training Set (%)'].append(train_mortality_5yr)
        mortality_risk_summary['Spatial Test Set (n)'].append(len(spatial_subset))
        mortality_risk_summary['Spatial Test Set (%)'].append(spatial_mortality_5yr)
        mortality_risk_summary['Temporal Test Set (n)'].append(len(temporal_subset))
        mortality_risk_summary['Temporal Test Set (%)'].append(temporal_mortality_5yr)
        
        # Add to HTML table data
        dialysis_html_data.append({
            'uacr_stage': uacr_stage,
            'egfr_stage': egfr_stage,
            'train_n': len(train_subset),
            'train_pct': train_dialysis_5yr,
            'spatial_n': len(spatial_subset),
            'spatial_pct': spatial_dialysis_5yr,
            'temporal_n': len(temporal_subset),
            'temporal_pct': temporal_dialysis_5yr
        })
        
        mortality_html_data.append({
            'uacr_stage': uacr_stage,
            'egfr_stage': egfr_stage,
            'train_n': len(train_subset),
            'train_pct': train_mortality_5yr,
            'spatial_n': len(spatial_subset),
            'spatial_pct': spatial_mortality_5yr,
            'temporal_n': len(temporal_subset),
            'temporal_pct': temporal_mortality_5yr
        })
```

### 5. Finalize and Save Plots

- Add legends to all plots
- Apply tight layout
- Save all figures

```python
# Add legends to all plots
# Discrete plots
handles_discrete, labels_discrete = axes_dialysis_discrete[0, 0].get_legend_handles_labels()
if handles_discrete:
    fig_dialysis_discrete.legend(handles_discrete, labels_discrete, loc='upper center', 
                               bbox_to_anchor=(0.5, 0.05), fancybox=True, shadow=True, ncol=3)
    fig_mortality_discrete.legend(handles_discrete, labels_discrete, loc='upper center', 
                                bbox_to_anchor=(0.5, 0.05), fancybox=True, shadow=True, ncol=3)

# Continuous plots
handles_continuous, labels_continuous = axes_dialysis_continuous[0, 0].get_legend_handles_labels()
if handles_continuous:
    fig_dialysis_continuous.legend(handles_continuous, labels_continuous, loc='upper center', 
                                 bbox_to_anchor=(0.5, 0.05), fancybox=True, shadow=True, ncol=3)
    fig_mortality_continuous.legend(handles_continuous, labels_continuous, loc='upper center', 
                                  bbox_to_anchor=(0.5, 0.05), fancybox=True, shadow=True, ncol=3)

# Save all figures
# Discrete plots
dialysis_discrete_path = os.path.join(output_path, 'dialysis_risk_yearly_points.png')
mortality_discrete_path = os.path.join(output_path, 'mortality_risk_yearly_points.png')

# Continuous plots
dialysis_continuous_path = os.path.join(output_path, 'dialysis_risk_continuous.png')
mortality_continuous_path = os.path.join(output_path, 'mortality_risk_continuous.png')

# Apply tight layout to all figures
fig_dialysis_discrete.tight_layout(rect=[0, 0.05, 1, 0.95])
fig_mortality_discrete.tight_layout(rect=[0, 0.05, 1, 0.95])
fig_dialysis_continuous.tight_layout(rect=[0, 0.05, 1, 0.95])
fig_mortality_continuous.tight_layout(rect=[0, 0.05, 1, 0.95])

# Save all figures
plt.figure(fig_dialysis_discrete.number)
plt.savefig(dialysis_discrete_path, dpi=300, bbox_inches='tight')

plt.figure(fig_mortality_discrete.number)
plt.savefig(mortality_discrete_path, dpi=300, bbox_inches='tight')

plt.figure(fig_dialysis_continuous.number)
plt.savefig(dialysis_continuous_path, dpi=300, bbox_inches='tight')

plt.figure(fig_mortality_continuous.number)
plt.savefig(mortality_continuous_path, dpi=300, bbox_inches='tight')

# Close all figures
plt.close(fig_dialysis_discrete)
plt.close(fig_mortality_discrete)
plt.close(fig_dialysis_continuous)
plt.close(fig_mortality_continuous)

visualization_paths.extend([
    dialysis_discrete_path, 
    mortality_discrete_path,
    dialysis_continuous_path,
    mortality_continuous_path
])
```

### 6. Create and Save CSV Summary Tables

- Create DataFrames from summary dictionaries
- Save to CSV files

```python
# Create CSV summary DataFrames
dialysis_summary_df = pd.DataFrame(dialysis_risk_summary)
mortality_summary_df = pd.DataFrame(mortality_risk_summary)

# Save CSV summary tables
dialysis_csv_path = os.path.join(output_path, 'dialysis_risk_summary.csv')
mortality_csv_path = os.path.join(output_path, 'mortality_risk_summary.csv')

dialysis_summary_df.to_csv(dialysis_csv_path, index=False)
mortality_summary_df.to_csv(mortality_csv_path, index=False)

visualization_paths.extend([dialysis_csv_path, mortality_csv_path])
```

### 7. Create and Save HTML Summary Tables

- Define function to create HTML table
- Create and save HTML tables

```python
# Function to create HTML table
def create_html_table(data, title, description):
    # Convert data to DataFrame
    df = pd.DataFrame(data)
    
    # Create pivot table with UACR stages as rows and eGFR stages as columns
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #2c3e50; }}
            p {{ color: #7f8c8d; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            th, td {{ padding: 12px; text-align: center; border: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; color: #333; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .dataset {{ font-weight: bold; }}
            .train {{ color: #3498db; }}
            .spatial {{ color: #2ecc71; }}
            .temporal {{ color: #e67e22; }}
            .small-sample {{ color: #999; font-style: italic; }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
        <p>{description}</p>
        <table>
            <tr>
                <th>UACR Stage / eGFR Stage</th>
    """
    
    # Add column headers (eGFR stages)
    for egfr_stage in egfr_stages:
        html_content += f"<th>{egfr_stage}</th>"
    
    html_content += "</tr>"
    
    # Add rows for each UACR stage
    for uacr_stage in uacr_stages:
        html_content += f"<tr><th>{uacr_stage}</th>"
        
        # Add cells for each eGFR stage
        for egfr_stage in egfr_stages:
            # Find the data for this combination
            cell_data = [item for item in data if item['uacr_stage'] == uacr_stage and item['egfr_stage'] == egfr_stage]
            
            if cell_data:
                cell = cell_data[0]
                
                # Format the cell content with all three datasets
                train_class = "small-sample" if cell['train_n'] < 10 else ""
                spatial_class = "small-sample" if cell['spatial_n'] < 10 else ""
                temporal_class = "small-sample" if cell['temporal_n'] < 10 else ""
                
                train_pct = f"{cell['train_pct']:.1f}%" if not pd.isna(cell['train_pct']) else "N/A"
                spatial_pct = f"{cell['spatial_pct']:.1f}%" if not pd.isna(cell['spatial_pct']) else "N/A"
                temporal_pct = f"{cell['temporal_pct']:.1f}%" if not pd.isna(cell['temporal_pct']) else "N/A"
                
                html_content += f"""
                <td>
                    <div class="dataset train {train_class}">Train: {train_pct} (n={cell['train_n']})</div>
                    <div class="dataset spatial {spatial_class}">Spatial: {spatial_pct} (n={cell['spatial_n']})</div>
                    <div class="dataset temporal {temporal_class}">Temporal: {temporal_pct} (n={cell['temporal_n']})</div>
                </td>
                """
            else:
                html_content += "<td>No data</td>"
        
        html_content += "</tr>"
    
    html_content += """
        </table>
        <p><em>Note: Values in lighter text indicate groups with fewer than 10 patients.</em></p>
    </body>
    </html>
    """
    
    return html_content

# Create and save HTML tables
dialysis_html = create_html_table(
    dialysis_html_data,
    "5-Year Risk of Dialysis (Event 1) by UACR and eGFR Stage",
    "Cumulative incidence (%) of dialysis initiation at 5 years across different UACR and eGFR stages."
)

mortality_html = create_html_table(
    mortality_html_data,
    "5-Year Risk of All-Cause Mortality (Event 2) by UACR and eGFR Stage",
    "Cumulative incidence (%) of all-cause mortality at 5 years across different UACR and eGFR stages."
)

# Save HTML tables
dialysis_html_path = os.path.join(output_path, 'dialysis_risk_summary.html')
mortality_html_path = os.path.join(output_path, 'mortality_risk_summary.html')

with open(dialysis_html_path, 'w') as f:
    f.write(dialysis_html)

with open(mortality_html_path, 'w') as f:
    f.write(mortality_html)

visualization_paths.extend([dialysis_html_path, mortality_html_path])
```

### 8. Return Results

- Return dictionary with paths to all generated files

```python
# Return dictionary with paths
return {
    "visualization_paths": visualization_paths,
    "dialysis_summary_csv_path": dialysis_csv_path,
    "mortality_summary_csv_path": mortality_csv_path,
    "dialysis_summary_html_path": dialysis_html_path,
    "mortality_summary_html_path": mortality_html_path
}
```

## Integration with perform_eda Function

The `perform_eda` function needs to be modified to call our new `compare_event_risks` function:

```python
@step(enable_cache=False)
def perform_eda(
    train_df: pd.DataFrame,
    temporal_test_df: pd.DataFrame,
    spatial_test_df: pd.DataFrame,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    # ... existing code ...
    
    # Generate visualizations
    visualization_paths = generate_visualizations(train_first, temporal_first, spatial_first, output_path)
    
    # Add new function call to compare event risks
    event_risk_results = compare_event_risks(train_df, temporal_test_df, spatial_test_df, output_path)
    visualization_paths.extend(event_risk_results["visualization_paths"])
    
    # Return results
    results = {
        "stats_table_path": stats_path,
        "visualization_paths": visualization_paths,
        "dialysis_summary_csv_path": event_risk_