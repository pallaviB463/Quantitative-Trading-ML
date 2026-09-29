from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# 1. Load the processed dataset
df = pd.read_parquet(
    "data/processed/cleaned_dataset.parquet"
)

df = df.sort_values("Date").reset_index(drop=True)

# 2. Select target and features
target = "Target_Return_1d"

feature_cols = [
    col for col in df.columns
    if col.startswith("Return_Delta")
]

# 3. Reproduce the chronological split
n = len(df)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

gap = 20

train_df = df.iloc[:train_end - gap]
val_df = df.iloc[train_end:val_end - gap]

# 4. Select features using training data only
available_features = [
    col for col in feature_cols
    if train_df[col].notna().any()
]

X_train = train_df[available_features]
y_train = train_df[target]

X_val = val_df[available_features]
y_val = val_df[target]

# 5. Hyperparameters to test
max_depths = [2, 3, 5, 7]
min_samples_leaf_values = [10, 20, 50, 100]

results = []

# 6. Train and evaluate each combination
for depth in max_depths:
    for leaf_size in min_samples_leaf_values:

        model = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "regressor",
                DecisionTreeRegressor(
                    max_depth=depth,
                    min_samples_leaf=leaf_size,
                    random_state=42
                )
            )
        ])

        model.fit(X_train, y_train)

        predictions = model.predict(X_val)

        rmse = np.sqrt(
            mean_squared_error(y_val, predictions)
        )

        results.append({
            "max_depth": depth,
            "min_samples_leaf": leaf_size,
            "RMSE": rmse
        })

        print(
            f"depth={depth}, "
            f"min_samples_leaf={leaf_size}, "
            f"RMSE={rmse:.6f}"
        )

# 7. Sort and save the results
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "RMSE"
).reset_index(drop=True)

output_dir = Path("results/tables")
output_dir.mkdir(parents=True, exist_ok=True)

results_df.to_csv(
    output_dir / "decision_tree_tuning.csv",
    index=False
)

# 8. Display the best configuration
print("\nBest Decision Tree configuration:")
print(results_df.iloc[0])

# 9. Compare against the baseline
baseline_rmse = np.sqrt(
    mean_squared_error(
        y_val,
        np.zeros(len(y_val))
    )
)

print("\nZero-return baseline RMSE:", baseline_rmse)