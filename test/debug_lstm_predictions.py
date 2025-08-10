import pandas as pd
import numpy as np

# Load the temporal test data
temporal_df = pd.read_csv('data/processed_temporal_test.csv')
print("Temporal dataset shape:", temporal_df.shape)
print("Unique patients in temporal:", temporal_df['key'].nunique())

# Check how many observations per patient
patient_counts = temporal_df['key'].value_counts()
print("\nObservations per patient stats:")
print(f"Min: {patient_counts.min()}")
print(f"Max: {patient_counts.max()}")
print(f"Mean: {patient_counts.mean():.2f}")

# Show example of patient with multiple observations
example_patient = patient_counts.index[0]
patient_data = temporal_df[temporal_df['key'] == example_patient].sort_values('date')
print(f"\nExample patient {example_patient} has {len(patient_data)} observations:")
print(patient_data[['key', 'date', 'duration', 'endpoint']].head())

# Check if we need to predict for each row or just the last observation per patient
print("\nDo all rows for a patient have the same duration and endpoint?")
for pid in temporal_df['key'].unique()[:5]:  # Check first 5 patients
    patient_rows = temporal_df[temporal_df['key'] == pid]
    unique_durations = patient_rows['duration'].nunique()
    unique_endpoints = patient_rows['endpoint'].nunique()
    print(f"Patient {pid}: {len(patient_rows)} rows, {unique_durations} unique durations, {unique_endpoints} unique endpoints")