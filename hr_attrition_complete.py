"""
HR Employee Attrition Analysis - Complete Implementation
=========================================================
This script performs comprehensive HR analytics including:
- Data loading and exploration
- Data cleaning and preprocessing
- Database storage (SQLite)
- Machine Learning model training (Random Forest)
- Model evaluation and feature importance
- Predictive analytics

For Streamlit dashboard, see the separate dashboard file.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sqlalchemy import create_engine

# ============================================
# PART 1: DATA LOADING AND EXPLORATION
# ============================================

print("=" * 60)
print("PART 1: DATA LOADING AND EXPLORATION")
print("=" * 60)

# Load dataset
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Check first 5 rows
print("\n📊 First 5 rows of the dataset:")
print(df.head())

# Basic info
print("\n📋 Dataset Information:")
df.info()

# Summary statistics
print("\n📈 Summary Statistics:")
print(df.describe())

# Check missing values
print("\n🔍 Missing Values Check:")
print(df.isnull().sum())

# ============================================
# PART 2: DATA CLEANING AND PREPROCESSING
# ============================================

print("\n" + "=" * 60)
print("PART 2: DATA CLEANING AND PREPROCESSING")
print("=" * 60)

# Drop duplicates
initial_count = len(df)
df.drop_duplicates(inplace=True)
print(f"\n🧹 Removed {initial_count - len(df)} duplicate rows")

# Convert 'Attrition' and 'OverTime' (Yes/No) to numeric 1/0
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
df['OverTime'] = df['OverTime'].map({'Yes': 1, 'No': 0})
print("✅ Converted 'Attrition' and 'OverTime' to binary (1/0)")

# Convert all other object columns to 'category' type
for col in df.select_dtypes('object').columns:
    df[col] = df[col].astype('category')
print(f"✅ Converted {len(df.select_dtypes('category').columns)} columns to category type")

# Correlation between selected features (numeric only)
print("\n📊 Correlation Matrix (Selected Features):")
corr_features = ['Attrition', 'Age', 'MonthlyIncome', 'JobSatisfaction', 'OverTime', 'YearsAtCompany']
correlation_matrix = df[corr_features].corr()
print(correlation_matrix)

# Save cleaned dataset
df.to_csv("cleaned_hr_data.csv", index=False)
print("\n💾 Cleaned dataset saved as 'cleaned_hr_data.csv'")

# ============================================
# PART 3: DATABASE OPERATIONS
# ============================================

print("\n" + "=" * 60)
print("PART 3: DATABASE OPERATIONS")
print("=" * 60)

# Create SQLite database and load data
engine = create_engine("sqlite:///workforce.db")
df.to_sql("employee_attrition", con=engine, if_exists="replace", index=False)
print("✅ Data loaded into SQLite database: workforce.db")
print(f"   - Table: employee_attrition")
print(f"   - Records: {len(df)}")

# ============================================
# PART 4: MACHINE LEARNING MODEL TRAINING
# ============================================

print("\n" + "=" * 60)
print("PART 4: MACHINE LEARNING MODEL TRAINING")
print("=" * 60)

# Reload dataset for modeling (fresh start)
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Convert target variable
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# Select features
features = [
    'Age', 'MonthlyIncome', 'DistanceFromHome',
    'JobSatisfaction', 'YearsAtCompany', 'OverTime',
    'WorkLifeBalance', 'EnvironmentSatisfaction'
]
X = df[features]
y = df['Attrition']

print(f"\n📋 Selected Features: {len(features)}")
for feature in features:
    print(f"   - {feature}")

# Encode categorical variables
X = pd.get_dummies(X, drop_first=True)
print(f"\n✅ After encoding: {X.shape[1]} features")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n📊 Data Split:")
print(f"   - Training set: {len(X_train)} samples")
print(f"   - Test set: {len(X_test)} samples")
print(f"   - Attrition rate in training: {y_train.mean():.2%}")
print(f"   - Attrition rate in test: {y_test.mean():.2%}")

# Train Random Forest model
print("\n🤖 Training Random Forest Classifier...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train, y_train)
print("✅ Model training completed!")

# ============================================
# PART 5: MODEL EVALUATION
# ============================================

print("\n" + "=" * 60)
print("PART 5: MODEL EVALUATION")
print("=" * 60)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"\n🎯 Accuracy: {accuracy:.2%}")
print("\n📊 Confusion Matrix:")
print(conf_matrix)
print("\n📈 Classification Report:")
print(class_report)

# Feature importance
print("\n⭐ Top 10 Most Important Features:")
feat_imp = pd.Series(model.feature_importances_, index=X.columns)
top_features = feat_imp.nlargest(10)
for idx, (feature, importance) in enumerate(top_features.items(), 1):
    print(f"   {idx:2d}. {feature:30s} - {importance:.4f}")

# Visualize feature importance
plt.figure(figsize=(10, 6))
top_features.sort_values().plot(kind='barh', color='steelblue')
plt.title("Top 10 Features Influencing Employee Attrition", fontsize=14, fontweight='bold')
plt.xlabel("Importance Score", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("\n💾 Feature importance chart saved as 'feature_importance.png'")
plt.show()

# ============================================
# PART 6: MODEL PERSISTENCE
# ============================================

print("\n" + "=" * 60)
print("PART 6: MODEL PERSISTENCE")
print("=" * 60)

# Save model
joblib.dump(model, "attrition_model.pkl")
print("✅ Model saved as 'attrition_model.pkl'")

# Save feature names for future predictions
joblib.dump(list(X.columns), "model_features.pkl")
print("✅ Feature names saved as 'model_features.pkl'")

# ============================================
# PART 7: TEST PREDICTIONS
# ============================================

print("\n" + "=" * 60)
print("PART 7: TEST PREDICTIONS")
print("=" * 60)

# Test prediction on multiple samples
print("\n🔮 Sample Predictions:")
for i in range(min(5, len(X_test))):
    sample = X_test.iloc[[i]]
    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0]
    
    print(f"\nSample {i+1}:")
    print(f"   Prediction: {'⚠️ Will Leave' if prediction == 1 else '✅ Will Stay'}")
    print(f"   Confidence: Stay={probability[0]:.2%}, Leave={probability[1]:.2%}")
    print(f"   Actual: {'Left' if y_test.iloc[i] == 1 else 'Stayed'}")

# ============================================
# SUMMARY
# ============================================

print("\n" + "=" * 60)
print("EXECUTION SUMMARY")
print("=" * 60)
print("✅ Data loaded and explored")
print("✅ Data cleaned and preprocessed")
print("✅ Data stored in SQLite database")
print("✅ Machine Learning model trained")
print("✅ Model evaluated and validated")
print("✅ Model saved for deployment")
print("\n📁 Generated Files:")
print("   - cleaned_hr_data.csv")
print("   - workforce.db")
print("   - attrition_model.pkl")
print("   - model_features.pkl")
print("   - feature_importance.png")
print("\n🎉 Analysis Complete!")
print("=" * 60)
