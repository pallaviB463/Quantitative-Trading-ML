from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_parquet(
    "data/processed/cleaned_dataset.parquet"
)

df = df.sort_values("Date").reset_index(drop=True)

target = "Target_Return_1d"

feature_cols = [
    col for col in df.columns
    if col.startswith("Return_Delta")
]


# --------------------------------------------------
# 2. Reproduce chronological split
# --------------------------------------------------

n = len(df)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

gap = 20

train_df = df.iloc[:train_end - gap]
val_df = df.iloc[train_end:val_end - gap]
test_df = df.iloc[val_end:]


# --------------------------------------------------
# 3. Training-only feature selection
# --------------------------------------------------

available_features = [
    col for col in feature_cols
    if train_df[col].notna().any()
]

X_train = train_df[available_features]
y_train = train_df[target]

X_test = test_df[available_features]
y_test = test_df[target]

print("Training samples:", len(X_train))
print("Validation samples:", len(val_df))
print("Testing samples:", len(X_test))
print("Selected features:", len(available_features))

print("\nTraining period:")
print(train_df["Date"].min(), "to", train_df["Date"].max())

print("\nTesting period:")
print(test_df["Date"].min(), "to", test_df["Date"].max())


# --------------------------------------------------
# 4. Models selected using validation results
# --------------------------------------------------

models = {
    "Elastic Net": Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "regressor",
            ElasticNet(
                alpha=0.1,
                l1_ratio=0.9,
                max_iter=20000,
                random_state=42
            )
        )
    ]),

    "Decision Tree": Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "regressor",
            DecisionTreeRegressor(
                max_depth=2,
                min_samples_leaf=10,
                random_state=42
            )
        )
    ]),

    "XGBoost": XGBRegressor(
        n_estimators=200,
        max_depth=2,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    ),

    "LightGBM": LGBMRegressor(
        n_estimators=200,
        max_depth=4,
        num_leaves=16,
        learning_rate=0.01,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
        n_jobs=-1
    )
}


# --------------------------------------------------
# 5. Baseline
# --------------------------------------------------

baseline_predictions = np.zeros(len(y_test))

baseline_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        baseline_predictions
    )
)

baseline_mae = mean_absolute_error(
    y_test,
    baseline_predictions
)

results = [{
    "Model": "Zero-return baseline",
    "RMSE": baseline_rmse,
    "MAE": baseline_mae
}]


# Store predictions for later backtesting
predictions_df = pd.DataFrame({
    "Date": test_df["Date"].values,
    "Actual_Return": y_test.values,
    "Zero_Return": baseline_predictions
})


# --------------------------------------------------
# 6. Train and test each model
# --------------------------------------------------

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    results.append({
        "Model": name,
        "RMSE": rmse,
        "MAE": mae
    })

    predictions_df[name] = predictions

    print(f"{name} Test RMSE: {rmse:.6f}")
    print(f"{name} Test MAE: {mae:.6f}")


# --------------------------------------------------
# 7. Save results
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "RMSE"
).reset_index(drop=True)

output_dir = Path("results/tables")
output_dir.mkdir(parents=True, exist_ok=True)

results_df.to_csv(
    output_dir / "final_test_results.csv",
    index=False
)

predictions_df.to_csv(
    output_dir / "test_predictions.csv",
    index=False
)


# --------------------------------------------------
# 8. Display final comparison
# --------------------------------------------------

print("\n================================")
print("FINAL TEST RESULTS")
print("================================")

print(results_df.to_string(index=False))

print("\nResults saved to:")
print("results/tables/final_test_results.csv")
print("results/tables/test_predictions.csv")