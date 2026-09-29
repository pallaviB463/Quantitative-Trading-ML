import pandas as pd
import numpy as np

# 1. Load the original dataset
df = pd.read_parquet(
    "data/raw/databaseFrankfurtComplete.parquet"
)

# 2. Convert and sort dates
df["Date"] = pd.to_datetime(df["Date"], utc=True)
df = df.sort_values("Date").reset_index(drop=True)

# 3. Select forecast horizons
horizons = [1, 5, 20]

target_cols = [
    f"Target_Return_{h}d"
    for h in horizons
]

# Calculate conventional forward returns
# directly from the closing prices.
for h in horizons:
    target_col = f"Target_Return_{h}d"

    df[target_col] = (
        df["Close"].shift(-h) / df["Close"] - 1
    )

# 4. Select historical return features
feature_cols = [
    col for col in df.columns
    if col.startswith("Return_Delta")
]

# 5. Keep dates, features and targets
selected_cols = (
    ["Date"] + feature_cols + target_cols
)

clean_df = df[selected_cols].copy()

# 6. Replace infinite values with NaN
clean_df = clean_df.replace(
    [np.inf, -np.inf],
    np.nan
)

# 7. Remove rows with missing targets
clean_df = clean_df.dropna(
    subset=target_cols
)

# 8. Save the prepared dataset
output_path = (
    "data/processed/cleaned_dataset.parquet"
)

clean_df.to_parquet(
    output_path,
    index=False
)

# 9. Display the results
print("Original dataset shape:", df.shape)

print(
    "Prepared dataset shape:",
    clean_df.shape
)

print(
    "Number of features:",
    len(feature_cols)
)

print(
    "Number of targets:",
    len(target_cols)
)

print(
    "Remaining infinite values:",
    np.isinf(
        clean_df[feature_cols].to_numpy(
            dtype=float
        )
    ).sum()
)

print(
    "Remaining missing feature values:",
    clean_df[feature_cols].isna().sum().sum()
)

print("\nDataset saved successfully!")

# 10. Analyze missing feature values

missing_percentage = (
    clean_df[feature_cols]
    .isna()
    .mean()
    * 100
)

print("\nMissing value analysis:")

print(
    "Features with more than 50% missing:",
    (missing_percentage > 50).sum()
)

print(
    "Features with more than 80% missing:",
    (missing_percentage > 80).sum()
)

print(
    "Features with 100% missing:",
    (missing_percentage == 100).sum()
)

print(
    "\nFeatures with no missing values:",
    (missing_percentage == 0).sum()
)

print("\nTop 10 features with most missing values:")

print(
    missing_percentage
    .sort_values(ascending=False)
    .head(10)
)