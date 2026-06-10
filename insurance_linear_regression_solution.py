# Predicting Medical Insurance Cost - 4Geeks Linear Regression Project
# Steps 1-4: Load data, EDA, build Linear Regression, optimize model

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------
# Step 1: Load dataset
# -----------------------------
DATA_PATH = "medical_insurance_cost.csv"

# If you keep the CSV in a data/raw folder, use this instead:
# DATA_PATH = "../data/raw/medical_insurance_cost.csv"

df = pd.read_csv(DATA_PATH)

print("First 5 rows:")
print(df.head())
print("\nDataset shape:", df.shape)
print("\nDataset info:")
print(df.info())
print("\nSummary statistics:")
print(df.describe(include="all"))


# -----------------------------
# Step 2: Full EDA
# -----------------------------
print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicated rows:", df.duplicated().sum())

print("\nCategorical value counts:")
for col in ["sex", "smoker", "region"]:
    print(f"\n{col}:")
    print(df[col].value_counts())

# Remove exact duplicate rows if any
# This dataset usually has 1 duplicate row.
df = df.drop_duplicates().reset_index(drop=True)

# Correlation for numeric variables
print("\nNumeric correlation with charges:")
print(df.corr(numeric_only=True)["charges"].sort_values(ascending=False))

# Basic EDA visualizations
plt.figure(figsize=(8, 5))
df["charges"].hist(bins=30)
plt.title("Distribution of Insurance Charges")
plt.xlabel("Charges")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(df["bmi"], df["charges"], alpha=0.5)
plt.title("BMI vs Insurance Charges")
plt.xlabel("BMI")
plt.ylabel("Charges")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
df.boxplot(column="charges", by="smoker")
plt.title("Charges by Smoker Status")
plt.suptitle("")
plt.xlabel("Smoker")
plt.ylabel("Charges")
plt.tight_layout()
plt.show()

# Features and target
X = df.drop("charges", axis=1)
y = df["charges"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

numeric_features = ["age", "bmi", "children"]
categorical_features = ["sex", "smoker", "region"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)
    ]
)


def evaluate_model(model_name, model, X_train, X_test, y_train, y_test):
    """Train model and print regression metrics."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n{model_name} Results")
    print("-" * 40)
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2:   {r2:.4f}")

    return model, y_pred, {"MAE": mae, "RMSE": rmse, "R2": r2}


# -----------------------------
# Step 3: Build Linear Regression model
# -----------------------------
linear_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]
)

linear_model, linear_predictions, linear_metrics = evaluate_model(
    "Default Linear Regression",
    linear_model,
    X_train,
    X_test,
    y_train,
    y_test
)


# -----------------------------
# Step 4: Optimize previous model
# -----------------------------
# Linear Regression cannot tune many hyperparameters, so we improve by:
# 1. Adding PolynomialFeatures to capture non-linear relationships
# 2. Using Ridge regularization to control overfitting

optimized_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
        ("model", Ridge(alpha=1.0))
    ]
)

optimized_model, optimized_predictions, optimized_metrics = evaluate_model(
    "Optimized Polynomial Ridge Regression",
    optimized_model,
    X_train,
    X_test,
    y_train,
    y_test
)

# Compare models
comparison = pd.DataFrame([linear_metrics, optimized_metrics], index=[
    "Default Linear Regression",
    "Optimized Polynomial Ridge Regression"
])
print("\nModel Comparison:")
print(comparison)

# Save the optimized model
os.makedirs("models", exist_ok=True)
joblib.dump(optimized_model, "models/optimized_insurance_model.pkl")
print("\nOptimized model saved to models/optimized_insurance_model.pkl")

# Optional: visualize actual vs predicted values
plt.figure(figsize=(8, 5))
plt.scatter(y_test, optimized_predictions, alpha=0.6)
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted Insurance Charges")
plt.tight_layout()
plt.show()
