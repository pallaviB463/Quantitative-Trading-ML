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

feature_cols = [
    col for col in df.columns
    if col.startswith("Return_Delta")
]

horizons = [1, 5, 20]

all_results = []
all_predictions = []

# --------------------------------------------------
# 2. Evaluate each forecast horizon
# --------------------------------------------------

for horizon in horizons:

    print("\n")
    print("=" * 60)
    print(f"FORECAST HORIZON: {horizon} DAY(S)")
    print("=" * 60)

    target = f"Target_Return_{horizon}d"

    # ----------------------------------------------
    # Chronological split
    # ----------------------------------------------

    n = len(df)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    # Prevent labels from crossing split boundaries
    gap = horizon

    train_df = df.iloc[:train_end - gap]
    val_df = df.iloc[
        train_end:val_end - gap
    ]
    test_df = df.iloc[val_end:]

    # ----------------------------------------------
    # Training-only feature selection
    # ----------------------------------------------

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
    print("Features:", len(available_features))

    # ----------------------------------------------
    # Selected models
    #
    # Hyperparameters were selected from our
    # previous validation experiments.
    # ----------------------------------------------

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

    # ----------------------------------------------
    # Zero-return baseline
    # ----------------------------------------------

    baseline_predictions = np.zeros(
        len(y_test)
    )

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

    all_results.append({
        "Horizon": horizon,
        "Model": "Zero-return baseline",
        "RMSE": baseline_rmse,
        "MAE": baseline_mae
    })

    print(
        f"\nZero-return baseline: "
        f"RMSE={baseline_rmse:.6f}, "
        f"MAE={baseline_mae:.6f}"
    )

    # ----------------------------------------------
    # Train models
    # ----------------------------------------------

    for name, model in models.items():

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )
        for date, actual, prediction in zip(
            test_df["Date"],
            y_test,
            predictions
        ):
            all_predictions.append({
                "Date": date,
                "Horizon": horizon,
                "Model": name,
                "Actual_Return": actual,
                "Predicted_Return": prediction
            })

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

        all_results.append({
            "Horizon": horizon,
            "Model": name,
            "RMSE": rmse,
            "MAE": mae
        })

        print(
            f"{name}: "
            f"RMSE={rmse:.6f}, "
            f"MAE={mae:.6f}"
        )


# --------------------------------------------------
# 3. Save complete results
# --------------------------------------------------

results_df = pd.DataFrame(
    all_results
)

output_dir = Path(
    "results/tables"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    output_dir /
    "multihorizon_test_results.csv",
    index=False
)


# --------------------------------------------------
# 4. Create RMSE comparison table
# --------------------------------------------------

rmse_table = results_df.pivot(
    index="Model",
    columns="Horizon",
    values="RMSE"
)

rmse_table.columns = [
    f"{col}_day_RMSE"
    for col in rmse_table.columns
]

rmse_table.to_csv(
    output_dir /
    "multihorizon_rmse_comparison.csv"
)
predictions_df = pd.DataFrame(
    all_predictions
)

predictions_df.to_csv(
    output_dir /
    "multihorizon_predictions.csv",
    index=False
)

# --------------------------------------------------
# 5. Print final comparison
# --------------------------------------------------

print("\n")
print("=" * 70)
print("MULTI-HORIZON TEST RMSE COMPARISON")
print("=" * 70)

print(
    rmse_table.to_string(
        float_format=lambda x: f"{x:.6f}"
    )
)

print("\nSaved:")
print(
    "results/tables/"
    "multihorizon_test_results.csv"
)

print(
    "results/tables/"
    "multihorizon_rmse_comparison.csv"
)