from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 1. Load the prepared dataset
df = pd.read_parquet(
    "data/processed/cleaned_dataset.parquet"
)

df = df.sort_values("Date").reset_index(drop=True)

# 2. Select the same target and features
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

# 5. Define the hyperparameters to test
alphas = [0.0001, 0.001, 0.01, 0.1]
l1_ratios = [0.1, 0.5, 0.9]

results = []

# 6. Train and evaluate each combination
for alpha in alphas:
    for l1_ratio in l1_ratios:

        model = Pipeline([
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
                    alpha=alpha,
                    l1_ratio=l1_ratio,
                    max_iter=20000,
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
            "alpha": alpha,
            "l1_ratio": l1_ratio,
            "RMSE": rmse
        })

        print(
            f"alpha={alpha}, "
            f"l1_ratio={l1_ratio}, "
            f"RMSE={rmse:.6f}"
        )

# 7. Save all tuning results
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "RMSE"
).reset_index(drop=True)

output_dir = Path("results/tables")
output_dir.mkdir(parents=True, exist_ok=True)

results_df.to_csv(
    output_dir / "elasticnet_tuning.csv",
    index=False
)

# 8. Display the best configuration
print("\nBest Elastic Net configuration:")
print(results_df.iloc[0])

# 9. Compare with the zero-return baseline
baseline_rmse = np.sqrt(
    mean_squared_error(
        y_val,
        np.zeros(len(y_val))
    )
)

print("\nZero-return baseline RMSE:", baseline_rmse)

# 10. Inspect the reported best Elastic Net model
best_alpha = results_df.iloc[0]["alpha"]
best_l1_ratio = results_df.iloc[0]["l1_ratio"]

best_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("regressor", ElasticNet(
        alpha=best_alpha,
        l1_ratio=best_l1_ratio,
        max_iter=20000,
        random_state=42
    ))
])

best_model.fit(X_train, y_train)

regressor = best_model.named_steps["regressor"]

nonzero_coefficients = np.count_nonzero(
    regressor.coef_
)

print("\nBest model inspection")
print("Alpha:", best_alpha)
print("L1 ratio:", best_l1_ratio)
print("Non-zero coefficients:", nonzero_coefficients)
print("Total coefficients:", len(regressor.coef_))
print("Intercept:", regressor.intercept_)

print("\nFirst 5 validation predictions:")
print(best_model.predict(X_val)[:5])