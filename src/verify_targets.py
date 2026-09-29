import pandas as pd

df = pd.read_parquet(
    "data/raw/databaseFrankfurtComplete.parquet"
)

df["Date"] = pd.to_datetime(df["Date"], utc=True)
df = df.sort_values("Date").reset_index(drop=True)

for horizon in [1, 5, 20]:
    target = f"Return_Future_Delta{horizon}_VW"

    # Formula 1: Conventional forward return
    forward_return = (
        df["Close"].shift(-horizon) / df["Close"] - 1
    )

    # Formula 2: Reverse return
    reverse_return = (
        df["Close"] / df["Close"].shift(-horizon) - 1
    )

    valid = (
        df[target].notna()
        & forward_return.notna()
        & reverse_return.notna()
    )

    forward_difference = (
        df.loc[valid, target]
        - forward_return.loc[valid]
    ).abs()

    reverse_difference = (
        df.loc[valid, target]
        - reverse_return.loc[valid]
    ).abs()

    print(f"\nHorizon: {horizon}")

    print(
        "Forward formula mean difference:",
        forward_difference.mean()
    )

    print(
        "Reverse formula mean difference:",
        reverse_difference.mean()
    )

    print(
        "Reverse formula maximum difference:",
        reverse_difference.max()
    )

    print("\nFirst 3 comparisons:")

    print(
        pd.DataFrame({
            "Stored target": df.loc[valid, target],
            "Forward return": forward_return.loc[valid],
            "Reverse return": reverse_return.loc[valid]
        }).head(3)
    )