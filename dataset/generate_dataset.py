"""
Dataset Generation Script for Student Academic Performance Prediction
This script generates a realistic synthetic dataset with 200 student records
"""

import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of records
n_records = 200

# Generate realistic data with correlations
# Study Hours: 1-10 hours per day
study_hours = np.random.uniform(1, 10, n_records)

# Attendance Percentage: 50-100%
attendance = np.random.uniform(50, 100, n_records)

# Previous Marks: 40-95
previous_marks = np.random.uniform(40, 95, n_records)

# Calculate Final Marks with realistic correlation
# Formula: Final Marks = base + weights * features + noise
# This creates a realistic relationship between inputs and output
final_marks = (
    20 +  # Base score
    study_hours * 3.5 +  # Study hours contribute significantly
    attendance * 0.4 +  # Attendance has moderate impact
    previous_marks * 0.3 +  # Previous performance matters
    np.random.normal(0, 5, n_records)  # Random noise for realism
)

# Ensure final marks are within realistic bounds (0-100)
final_marks = np.clip(final_marks, 0, 100)

# Create DataFrame
df = pd.DataFrame({
    'StudyHours': np.round(study_hours, 2),
    'Attendance': np.round(attendance, 2),
    'PreviousMarks': np.round(previous_marks, 2),
    'FinalMarks': np.round(final_marks, 2)
})

# Introduce some missing values (5% randomly) for preprocessing practice
missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
df.loc[missing_indices[:3], 'StudyHours'] = np.nan
df.loc[missing_indices[3:6], 'Attendance'] = np.nan
df.loc[missing_indices[6:9], 'PreviousMarks'] = np.nan

# Save dataset
df.to_csv('dataset/student_data.csv', index=False)

print("Dataset generated successfully!")
print(f"Total records: {len(df)}")
print("\nFirst 10 rows:")
print(df.head(10))
print("\nDataset Info:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())
print("\nMissing values:")
print(df.isnull().sum())
