"""
Machine Learning Model Training Script
=======================================
This script performs complete ML pipeline:
1. Data Loading
2. Data Preprocessing
3. Exploratory Data Analysis (EDA)
4. Model Training (Linear Regression)
5. Model Evaluation
6. Model Saving

Machine Learning Concepts Explained:
- Supervised Learning: Learning from labeled data (we have input-output pairs)
- Regression: Predicting continuous numerical values
- Linear Regression: Finding the best-fit line through data points
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

print("=" * 80)
print("STUDENT PERFORMANCE PREDICTION - MACHINE LEARNING MODEL TRAINING")
print("=" * 80)

# ============================================================================
# STEP 1: DATA LOADING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: DATA LOADING")
print("=" * 80)

try:
    df = pd.read_csv('dataset/student_data.csv')
    print(f"✓ Dataset loaded successfully!")
    print(f"✓ Total records: {len(df)}")
    print(f"✓ Total features: {len(df.columns)}")
    print(f"\nDataset Preview:")
    print(df.head(10))
except FileNotFoundError:
    print("✗ Error: Dataset not found. Please run 'python generate_dataset.py' first.")
    exit(1)

# ============================================================================
# STEP 2: DATA PREPROCESSING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: DATA PREPROCESSING")
print("=" * 80)

# Check for missing values
print("\nMissing Values Check:")
print("-" * 80)
missing_values = df.isnull().sum()
print(missing_values)

if missing_values.sum() > 0:
    print(f"\n⚠ Found {missing_values.sum()} missing values")
    print("➜ Handling Strategy: Filling missing values with column mean")
    df.fillna(df.mean(), inplace=True)
    print("✓ Missing values handled successfully!")
else:
    print("✓ No missing values found!")

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"\nDuplicate Records: {duplicates}")
if duplicates > 0:
    df.drop_duplicates(inplace=True)
    print("✓ Duplicates removed!")

# Detect outliers using IQR method
print("\nOutlier Detection (using IQR method):")
print("-" * 80)
for column in df.columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[column] < Q1 - 1.5 * IQR) | (df[column] > Q3 + 1.5 * IQR)]
    print(f"{column}: {len(outliers)} outliers detected")

print("\n✓ Data preprocessing completed!")

# ============================================================================
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: EXPLORATORY DATA ANALYSIS")
print("=" * 80)

# Statistical Summary
print("\nStatistical Summary:")
print("-" * 80)
print(df.describe())

# Correlation Analysis
print("\nCorrelation Matrix:")
print("-" * 80)
correlation_matrix = df.corr()
print(correlation_matrix)

# Create visualizations directory
os.makedirs('visualizations', exist_ok=True)

# Visualization 1: Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, fmt='.3f')
plt.title('Correlation Heatmap - Feature Relationships', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/01_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Correlation Heatmap")
plt.close()

# Visualization 2: Distribution of Features (Histograms)
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Distribution of Features', fontsize=16, fontweight='bold')

for idx, column in enumerate(df.columns):
    ax = axes[idx // 2, idx % 2]
    ax.hist(df[column], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    ax.set_title(f'{column} Distribution', fontweight='bold')
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/02_feature_distributions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Feature Distributions")
plt.close()

# Visualization 3: Box Plots (Outlier Detection)
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Box Plots - Outlier Detection', fontsize=16, fontweight='bold')

for idx, column in enumerate(df.columns):
    ax = axes[idx // 2, idx % 2]
    ax.boxplot(df[column], vert=True, patch_artist=True,
               boxprops=dict(facecolor='lightblue', alpha=0.7))
    ax.set_title(f'{column}', fontweight='bold')
    ax.set_ylabel('Values')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/03_boxplots.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Box Plots")
plt.close()

# Visualization 4: Scatter Plots (Feature vs Target)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Feature Relationships with Final Marks', fontsize=16, fontweight='bold')

features = ['StudyHours', 'Attendance', 'PreviousMarks']
colors = ['blue', 'green', 'red']

for idx, (feature, color) in enumerate(zip(features, colors)):
    axes[idx].scatter(df[feature], df['FinalMarks'], alpha=0.6, color=color, edgecolor='black')
    axes[idx].set_xlabel(feature, fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('Final Marks', fontsize=12, fontweight='bold')
    axes[idx].set_title(f'{feature} vs Final Marks', fontweight='bold')
    axes[idx].grid(alpha=0.3)
    
    # Add trend line
    z = np.polyfit(df[feature], df['FinalMarks'], 1)
    p = np.poly1d(z)
    axes[idx].plot(df[feature], p(df[feature]), "r--", alpha=0.8, linewidth=2)

plt.tight_layout()
plt.savefig('visualizations/04_scatter_plots.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Scatter Plots")
plt.close()

print("\n✓ Exploratory Data Analysis completed!")
print("✓ All visualizations saved in 'visualizations/' directory")

# ============================================================================
# STEP 4: MACHINE LEARNING - LINEAR REGRESSION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: MACHINE LEARNING MODEL TRAINING")
print("=" * 80)

# Prepare features (X) and target (y)
X = df[['StudyHours', 'Attendance', 'PreviousMarks']]
y = df['FinalMarks']

print("\nDataset Split Configuration:")
print("-" * 80)
print("➜ Training Set: 80% of data (used to train the model)")
print("➜ Testing Set: 20% of data (used to evaluate the model)")

# Split data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n✓ Training samples: {len(X_train)}")
print(f"✓ Testing samples: {len(X_test)}")

# Initialize Linear Regression model
print("\nInitializing Linear Regression Model...")
print("-" * 80)
print("Linear Regression Equation: y = w₁x₁ + w₂x₂ + w₃x₃ + b")
print("Where:")
print("  y = FinalMarks (prediction)")
print("  x₁ = StudyHours")
print("  x₂ = Attendance")
print("  x₃ = PreviousMarks")
print("  w₁, w₂, w₃ = Coefficients (weights)")
print("  b = Intercept (bias)")

model = LinearRegression()

# Train the model
print("\n➜ Training model on training data...")
model.fit(X_train, y_train)
print("✓ Model training completed!")

# Display model parameters
print("\nLearned Model Parameters:")
print("-" * 80)
print(f"Intercept (bias): {model.intercept_:.4f}")
print("\nCoefficients (weights):")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature}: {coef:.4f}")

print("\nFinal Model Equation:")
print(f"FinalMarks = {model.intercept_:.4f} + "
      f"({model.coef_[0]:.4f} × StudyHours) + "
      f"({model.coef_[1]:.4f} × Attendance) + "
      f"({model.coef_[2]:.4f} × PreviousMarks)")

# ============================================================================
# STEP 5: MODEL EVALUATION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: MODEL EVALUATION")
print("=" * 80)

# Make predictions on test data
y_pred = model.predict(X_test)

# Calculate evaluation metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\nModel Performance Metrics:")
print("-" * 80)
print(f"R² Score: {r2:.4f}")
print(f"  → Interpretation: Model explains {r2*100:.2f}% of variance in data")
print(f"  → Range: 0 to 1 (higher is better)")

print(f"\nMean Absolute Error (MAE): {mae:.4f}")
print(f"  → Interpretation: Average prediction error is ±{mae:.2f} marks")
print(f"  → Range: 0 to ∞ (lower is better)")

print(f"\nMean Squared Error (MSE): {mse:.4f}")
print(f"  → Interpretation: Penalizes larger errors more heavily")
print(f"  → Range: 0 to ∞ (lower is better)")

print(f"\nRoot Mean Squared Error (RMSE): {rmse:.4f}")
print(f"  → Interpretation: Standard deviation of prediction errors is {rmse:.2f} marks")
print(f"  → Range: 0 to ∞ (lower is better)")

# Performance Assessment
print("\nModel Performance Assessment:")
print("-" * 80)
if r2 > 0.9:
    print("✓ EXCELLENT: Model has very high predictive accuracy!")
elif r2 > 0.8:
    print("✓ GOOD: Model has good predictive accuracy")
elif r2 > 0.7:
    print("⚠ FAIR: Model has acceptable accuracy but can be improved")
else:
    print("✗ POOR: Model needs significant improvement")

# Visualization 5: Actual vs Predicted
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolor='black', s=100)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=3, label='Perfect Prediction')
plt.xlabel('Actual Final Marks', fontsize=14, fontweight='bold')
plt.ylabel('Predicted Final Marks', fontsize=14, fontweight='bold')
plt.title('Actual vs Predicted Final Marks', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/05_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Actual vs Predicted visualization")
plt.close()

# Visualization 6: Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.6, edgecolor='black')
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('Predicted Values', fontsize=14, fontweight='bold')
plt.ylabel('Residuals', fontsize=14, fontweight='bold')
plt.title('Residual Plot', fontsize=16, fontweight='bold')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/06_residual_plot.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Residual Plot")
plt.close()

# ============================================================================
# STEP 6: MODEL SAVING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: MODEL PERSISTENCE")
print("=" * 80)

# Create model directory
os.makedirs('model', exist_ok=True)

# Save model using joblib
model_path = 'model/student_model.pkl'
joblib.dump(model, model_path)
print(f"✓ Model saved successfully to: {model_path}")

# Test loading the model
loaded_model = joblib.load(model_path)
print("✓ Model loaded successfully for verification")

# ============================================================================
# STEP 7: SAMPLE PREDICTIONS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: SAMPLE PREDICTIONS")
print("=" * 80)

# Sample prediction examples
sample_data = [
    [6.5, 85, 75],  # Good student
    [3.0, 60, 50],  # Average student
    [9.0, 95, 90],  # Excellent student
]

sample_labels = ["Good Student", "Average Student", "Excellent Student"]

print("\nSample Predictions:")
print("-" * 80)
for data, label in zip(sample_data, sample_labels):
    prediction = loaded_model.predict([data])[0]
    print(f"\n{label}:")
    print(f"  Input: Study Hours={data[0]}, Attendance={data[1]}%, Previous Marks={data[2]}")
    print(f"  Predicted Final Marks: {prediction:.2f}")

print("\n" + "=" * 80)
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 80)
print("\n✓ Model is ready for deployment in Flask application")
print("✓ Run 'python app.py' to start the web application")
