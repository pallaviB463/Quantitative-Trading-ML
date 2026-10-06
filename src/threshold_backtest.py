from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. Load multi-horizon predictions
# --------------------------------------------------

df = pd.read_csv(
    "results/tables/multihorizon_predictions.csv"
)

df["Date"] = pd.to_datetime(
    df["Date"],
    utc=True
)

thresholds = [
    0.0000,   # 0%
    0.0025,   # 0.25%
    0.0050,   # 0.50%
    0.0100    # 1.00%
]

results = []


# --------------------------------------------------
# 2. Test every model / horizon / threshold
# --------------------------------------------------

for horizon in sorted(df["Horizon"].unique()):

    horizon_df = df[
        df["Horizon"] == horizon
    ]

    for model in horizon_df["Model"].unique():

        model_df = (
            horizon_df[
                horizon_df["Model"] == model
            ]
            .sort_values("Date")
            .reset_index(drop=True)
        )

        actual = model_df[
            "Actual_Return"
        ].values

        predicted = model_df[
            "Predicted_Return"
        ].values

        for threshold in thresholds:

            # --------------------------------------
            # Trading signal
            # --------------------------------------

            positions = np.where(
                predicted > threshold,
                1,
                np.where(
                    predicted < -threshold,
                    -1,
                    0
                )
            )

            # --------------------------------------
            # Strategy return
            # --------------------------------------

            strategy_returns = (
                positions * actual
            )

            cumulative = np.prod(
                1 + strategy_returns
            )

            total_return = cumulative - 1

            # --------------------------------------
            # Directional accuracy
            #
            # Only evaluate days where a position
            # was actually taken.
            # --------------------------------------

            active = positions != 0

            number_of_active_days = (
                active.sum()
            )

            if number_of_active_days > 0:

                directional_accuracy = (
                    np.sign(
                        positions[active]
                    )
                    ==
                    np.sign(
                        actual[active]
                    )
                ).mean()

            else:
                directional_accuracy = np.nan

            # --------------------------------------
            # Position changes
            # --------------------------------------

            position_changes = (
                pd.Series(positions)
                .diff()
                .fillna(positions[0])
                .ne(0)
                .sum()
            )

            # --------------------------------------
            # Save result
            # --------------------------------------

            results.append({
                "Horizon": horizon,
                "Model": model,
                "Threshold": threshold,
                "Total_Return": total_return,
                "Active_Days": number_of_active_days,
                "Directional_Accuracy":
                    directional_accuracy,
                "Position_Changes":
                    position_changes
            })


# --------------------------------------------------
# 3. Create results dataframe
# --------------------------------------------------

results_df = pd.DataFrame(
    results
)

results_df["Threshold_Percent"] = (
    results_df["Threshold"] * 100
)


# --------------------------------------------------
# 4. Save full results
# --------------------------------------------------

output_dir = Path(
    "results/tables"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    output_dir /
    "threshold_backtest_results.csv",
    index=False
)


# --------------------------------------------------
# 5. Print results by horizon
# --------------------------------------------------

for horizon in sorted(
    results_df["Horizon"].unique()
):

    print("\n")
    print("=" * 75)

    print(
        f"{horizon}-DAY THRESHOLD "
        "TRADING RESULTS"
    )

    print("=" * 75)

    display_df = results_df[
        results_df["Horizon"] == horizon
    ][
        [
            "Model",
            "Threshold_Percent",
            "Total_Return",
            "Active_Days",
            "Directional_Accuracy",
            "Position_Changes"
        ]
    ]

    print(
        display_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )


# --------------------------------------------------
# 6. Find best threshold for each model/horizon
# --------------------------------------------------

best_results = (
    results_df
    .sort_values(
        "Total_Return",
        ascending=False
    )
    .groupby(
        ["Horizon", "Model"],
        as_index=False
    )
    .first()
)

best_results.to_csv(
    output_dir /
    "best_threshold_results.csv",
    index=False
)

print("\n")
print("=" * 75)
print("BEST THRESHOLD FOR EACH MODEL / HORIZON")
print("=" * 75)

print(
    best_results[
        [
            "Horizon",
            "Model",
            "Threshold_Percent",
            "Total_Return",
            "Active_Days",
            "Directional_Accuracy"
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)