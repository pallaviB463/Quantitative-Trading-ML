import pandas as pd

# --------------------------------------------------
# 1. Load RAW dataset because it contains Close
# --------------------------------------------------

df = pd.read_parquet(
    "data/raw/databaseFrankfurtComplete.parquet"
)

df["Date"] = pd.to_datetime(
    df["Date"],
    utc=True
)

df = df.sort_values("Date").reset_index(drop=True)


# --------------------------------------------------
# 2. Load saved test predictions
# --------------------------------------------------

pred = pd.read_csv(
    "results/tables/test_predictions.csv"
)

pred["Date"] = pd.to_datetime(
    pred["Date"],
    utc=True
)


# --------------------------------------------------
# 3. Display first predictions
# --------------------------------------------------

print("\nFirst 5 prediction rows:")

print(
    pred[
        [
            "Date",
            "Actual_Return",
            "Elastic Net",
            "Decision Tree",
            "XGBoost",
            "LightGBM"
        ]
    ].head()
)


# --------------------------------------------------
# 4. Verify target alignment
# --------------------------------------------------

print("\nChecking actual return calculation:")

for i in range(5):

    date = pred.iloc[i]["Date"]

    row_index = df.index[
        df["Date"] == date
    ]

    if len(row_index) == 0:
        print("Date not found:", date)
        continue

    idx = row_index[0]

    current_close = df.loc[idx, "Close"]
    next_close = df.loc[idx + 1, "Close"]

    calculated_return = (
        next_close / current_close - 1
    )

    stored_return = pred.iloc[i]["Actual_Return"]

    print("\nDate:", date)
    print("Current close:", current_close)
    print("Next close:", next_close)

    print(
        "Calculated t -> t+1 return:",
        calculated_return
    )

    print(
        "Stored Actual_Return:",
        stored_return
    )

    print(
        "Difference:",
        abs(
            calculated_return -
            stored_return
        )
    )