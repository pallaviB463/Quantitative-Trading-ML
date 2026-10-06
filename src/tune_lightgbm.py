from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor

# 1. Load dataset
df = pd.read_parquet(
    "data/processed/cleaned_dataset.parquet"
)

df = df.sort_values("Date").reset_index(drop=True)

# 2. Target and features
target = "Target_Return_1d"

feature_cols = [
    col for col in df.columns
    if col.startswith("Return_Delta")
]

# 3. Chronological split
n = len(df)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

gap = 20

train_df = df.iloc[:train_end - gap]
val_df = df.iloc[train_end:val_end - gap]

# 4. Training-only feature selection
available_features = [
    col for col in feature_cols
    if train_df[col].notna().any()
]

X_train = train_df[available_features]
y_train = train_df[target]

X_val = val_df[available_features]
y_val = val_df[target]

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Selected features:", len(available_features))

# 5. Hyperparameters
max_depths = [2, 3, 4]
learning_rates = [0.01, 0.05, 0.1]

results = []

# 6. Train configurations
for depth in max_depths:
    for learning_rate in learning_rates:

        # Keep number of leaves compatible with depth
        num_leaves = min(2 ** depth, 31)

        model = LGBMRegressor(
            n_estimators=200,
            max_depth=depth,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_val)

        rmse = np.sqrt(
            mean_squared_error(y_val, predictions)
        )

        results.append({
            "max_depth": depth,
            "num_leaves": num_leaves,
            "learning_rate": learning_rate,
            "RMSE": rmse
        })

        print(
            f"depth={depth}, "
            f"num_leaves={num_leaves}, "
            f"learning_rate={learning_rate}, "
            f"RMSE={rmse:.6f}"
        )

# 7. Sort results
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "RMSE"
).reset_index(drop=True)

# 8. Save results
output_dir = Path("results/tables")
output_dir.mkdir(parents=True, exist_ok=True)

results_df.to_csv(
    output_dir / "lightgbm_tuning.csv",
    index=False
)

# 9. Best configuration
print("\nBest LightGBM configuration:")
print(results_df.iloc[0])

# 10. Baseline
baseline_rmse = np.sqrt(
    mean_squared_error(
        y_val,
        np.zeros(len(y_val))
    )
)

print("\nZero-return baseline RMSE:", baseline_rmse)