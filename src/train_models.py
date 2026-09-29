import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.linear_model import ElasticNet
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1. Load the prepared dataset
df = pd.read_parquet(
    "data/processed/cleaned_dataset.parquet"
)

df = df.sort_values("Date").reset_index(drop=True)

# 2. Select the prediction target
horizon = 1

target = f"Target_Return_{horizon}d"

# 3. Select historical features
feature_cols = [
    col for col in df.columns
    if col.startswith("Return_Delta")
]

# 4. Create chronological splits
n = len(df)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

# Use the maximum forecast horizon as the gap
gap = 20

train_df = df.iloc[:train_end - gap]
val_df = df.iloc[train_end:val_end - gap]
test_df = df.iloc[val_end:]

print("\nDate ranges:")
print("Training:", train_df["Date"].min(), "to", train_df["Date"].max())
print("Validation:", val_df["Date"].min(), "to", val_df["Date"].max())
print("Testing:", test_df["Date"].min(), "to", test_df["Date"].max())

# 5. Select features using training data only
available_features = [
    col for col in feature_cols
    if train_df[col].notna().any()
]

X_train = train_df[available_features]
X_val = val_df[available_features]
X_test = test_df[available_features]

y_train = train_df[target]
y_val = val_df[target]
y_test = test_df[target]

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples:", len(X_test))
print("Selected features:", len(available_features))

# 6. Create Elastic Net pipeline
model = Pipeline([
    ("imputer", SimpleImputer(
        strategy="median"
    )),
    ("scaler", StandardScaler()),
    ("regressor", ElasticNet(
        alpha=0.001,
        l1_ratio=0.5,
        max_iter=10000,
        random_state=42
    ))
])

# 7. Train the model
model.fit(X_train, y_train)

# 8. Predict validation returns
val_predictions = model.predict(X_val)

# 9. Calculate validation metrics
rmse = np.sqrt(
    mean_squared_error(y_val, val_predictions)
)

mae = mean_absolute_error(
    y_val,
    val_predictions
)

print("\nElastic Net Validation Results")
print("RMSE:", rmse)
print("MAE:", mae)

# 10. Zero-return baseline
baseline_predictions = np.zeros(len(y_val))

baseline_rmse = np.sqrt(
    mean_squared_error(y_val, baseline_predictions)
)

baseline_mae = mean_absolute_error(
    y_val, baseline_predictions
)

print("\nZero-Return Baseline Results")
print("RMSE:", baseline_rmse)
print("MAE:", baseline_mae)

# 11. Compare the model with the baseline
rmse_improvement = (
    (baseline_rmse - rmse) / baseline_rmse
) * 100

print("\nComparison")
print(
    "Elastic Net RMSE improvement (%):",
    rmse_improvement
)

# 12. Decision Tree pipeline
tree_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("regressor", DecisionTreeRegressor(
        max_depth=5,
        min_samples_leaf=20,
        random_state=42
    ))
])

# 13. Train the Decision Tree
tree_model.fit(X_train, y_train)

# 14. Predict validation returns
tree_predictions = tree_model.predict(X_val)

# 15. Calculate validation metrics
tree_rmse = np.sqrt(
    mean_squared_error(y_val, tree_predictions)
)

tree_mae = mean_absolute_error(
    y_val, tree_predictions
)

print("\nDecision Tree Validation Results")
print("RMSE:", tree_rmse)
print("MAE:", tree_mae)

# 16. Compare all three approaches
print("\nRMSE Comparison")
print("Zero-return baseline:", baseline_rmse)
print("Elastic Net:", rmse)
print("Decision Tree:", tree_rmse)

# 17. XGBoost model
xgb_model = XGBRegressor(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

# 18. Train XGBoost
# XGBoost handles missing feature values natively.
xgb_model.fit(X_train, y_train)

# 19. Predict validation returns
xgb_predictions = xgb_model.predict(X_val)

# 20. Calculate validation metrics
xgb_rmse = np.sqrt(
    mean_squared_error(y_val, xgb_predictions)
)

xgb_mae = mean_absolute_error(
    y_val, xgb_predictions
)

print("\nXGBoost Validation Results")
print("RMSE:", xgb_rmse)
print("MAE:", xgb_mae)

# 21. Compare all approaches
print("\nRMSE Comparison")
print("Zero-return baseline:", baseline_rmse)
print("Elastic Net:", rmse)
print("Decision Tree:", tree_rmse)
print("XGBoost:", xgb_rmse)

# 22. LightGBM model
lgbm_model = LGBMRegressor(
    n_estimators=200,
    max_depth=3,
    num_leaves=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=-1,
    n_jobs=-1
)

# 23. Train LightGBM
lgbm_model.fit(X_train, y_train)

# 24. Predict validation returns
lgbm_predictions = lgbm_model.predict(X_val)

# 25. Calculate validation metrics
lgbm_rmse = np.sqrt(
    mean_squared_error(y_val, lgbm_predictions)
)

lgbm_mae = mean_absolute_error(
    y_val, lgbm_predictions
)

print("\nLightGBM Validation Results")
print("RMSE:", lgbm_rmse)
print("MAE:", lgbm_mae)

# 26. Compare all approaches
print("\nFinal Validation RMSE Comparison")
print("Zero-return baseline:", baseline_rmse)
print("Elastic Net:", rmse)
print("Decision Tree:", tree_rmse)
print("XGBoost:", xgb_rmse)
print("LightGBM:", lgbm_rmse)

# 27. Save validation results
from pathlib import Path

results = pd.DataFrame({
    "Model": [
        "Zero-return baseline",
        "Elastic Net",
        "Decision Tree",
        "XGBoost",
        "LightGBM"
    ],
    "RMSE": [
        baseline_rmse,
        rmse,
        tree_rmse,
        xgb_rmse,
        lgbm_rmse
    ],
    "MAE": [
        baseline_mae,
        mae,
        tree_mae,
        xgb_mae,
        lgbm_mae
    ]
})

output_dir = Path("results/tables")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "validation_results.csv"

results.to_csv(output_file, index=False)

print("\nValidation results saved to:")
print(output_file)