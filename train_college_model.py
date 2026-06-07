"""
College Placement Prediction - ML Model Training
================================================
Trains multiple ML models for:
1. Placement Probability Prediction
2. Package Prediction

Uses advanced features and ensemble methods for high accuracy.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("COLLEGE PLACEMENT PREDICTION - MODEL TRAINING")
print("=" * 80)

# Load dataset
print("\n[1/6] Loading dataset...")
try:
    df = pd.read_csv('dataset/college_students_data.csv')
    print(f"✓ Dataset loaded: {len(df)} students, {len(df.columns)} features")
except:
    print("✗ Error: Dataset not found. Run 'python generate_dataset.py' first!")
    exit(1)

# Handle missing values
print("\n[2/6] Preprocessing data...")
df.fillna(df.mean(numeric_only=True), inplace=True)
print("✓ Missing values handled")

# Select features for training
feature_columns = [
    'StudyHours', 'Attendance', 'CGPA', 'MathMarks', 'ProgrammingMarks', 'TheoryMarks',
    'DSAScore', 'CodingScore', 'WebDevScore', 'LeetCodeSolved', 'CodeChefRating',
    'JavaProficiency', 'PythonProficiency', 'CppProficiency', 'JavaScriptProficiency',
    'FrontendSkills', 'BackendSkills', 'DatabaseSkills',
    'ProjectsCount', 'GitHubRepos', 'InternshipsCount',
    'CommunicationSkill', 'AptitudeScore', 'ReasoningScore', 
    'MockTestScore', 'InterviewPrepHours', 'ResumeScore'
]

X = df[feature_columns]
y_placement = df['PlacementProbability']
y_package = df['ExpectedPackage']

print(f"✓ Features selected: {len(feature_columns)}")
print(f"✓ Target 1: Placement Probability")
print(f"✓ Target 2: Expected Package")

# Split data
print("\n[3/6] Splitting data (80% train, 20% test)...")
X_train, X_test, y_place_train, y_place_test = train_test_split(
    X, y_placement, test_size=0.2, random_state=42
)
_, _, y_pkg_train, y_pkg_test = train_test_split(
    X, y_package, test_size=0.2, random_state=42
)
print(f"✓ Training samples: {len(X_train)}")
print(f"✓ Testing samples: {len(X_test)}")

# Train Placement Probability Model
print("\n[4/6] Training Placement Probability Model...")
print("  Algorithm: Random Forest Regressor")
placement_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
placement_model.fit(X_train, y_place_train)
y_place_pred = placement_model.predict(X_test)

# Evaluate placement model
place_r2 = r2_score(y_place_test, y_place_pred)
place_mae = mean_absolute_error(y_place_test, y_place_pred)
place_rmse = np.sqrt(mean_squared_error(y_place_test, y_place_pred))

print(f"✓ Model trained successfully!")
print(f"  R² Score: {place_r2:.4f} ({place_r2*100:.2f}% accuracy)")
print(f"  MAE: {place_mae:.2f}%")
print(f"  RMSE: {place_rmse:.2f}%")

# Train Package Prediction Model
print("\n[5/6] Training Package Prediction Model...")
print("  Algorithm: Gradient Boosting Regressor")
package_model = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=7,
    learning_rate=0.1,
    random_state=42
)
package_model.fit(X_train, y_pkg_train)
y_pkg_pred = package_model.predict(X_test)

# Evaluate package model
pkg_r2 = r2_score(y_pkg_test, y_pkg_pred)
pkg_mae = mean_absolute_error(y_pkg_test, y_pkg_pred)
pkg_rmse = np.sqrt(mean_squared_error(y_pkg_test, y_pkg_pred))

print(f"✓ Model trained successfully!")
print(f"  R² Score: {pkg_r2:.4f} ({pkg_r2*100:.2f}% accuracy)")
print(f"  MAE: {pkg_mae:.2f} LPA")
print(f"  RMSE: {pkg_rmse:.2f} LPA")

# Feature importance
print("\n[6/6] Analyzing feature importance...")
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': placement_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print("-" * 80)
for idx, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']:30s} {row['importance']:.4f}")

# Save models
os.makedirs('model', exist_ok=True)
model_data = {
    'placement_model': placement_model,
    'package_model': package_model,
    'feature_columns': feature_columns,
    'placement_r2': place_r2,
    'package_r2': pkg_r2,
    'feature_importance': feature_importance
}

joblib.dump(model_data, 'model/college_placement_model.pkl')
print("\n" + "=" * 80)
print("✓ Models saved to: model/college_placement_model.pkl")
print("=" * 80)

# Sample predictions
print("\nSample Predictions:")
print("-" * 80)
samples = X_test.head(5)
place_samples = placement_model.predict(samples)
pkg_samples = package_model.predict(samples)

for i in range(5):
    print(f"\nStudent {i+1}:")
    print(f"  CGPA: {samples.iloc[i]['CGPA']:.2f}, DSA: {samples.iloc[i]['DSAScore']:.0f}, Projects: {samples.iloc[i]['ProjectsCount']:.0f}")
    print(f"  → Placement Prob: {place_samples[i]:.1f}%, Package: {pkg_samples[i]:.2f} LPA")

print("\n" + "=" * 80)
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 80)
print("\n✓ Run 'python app_college.py' to start the web application")
