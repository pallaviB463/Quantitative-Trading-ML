from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. Load predictions
# --------------------------------------------------

df = pd.read_csv(
    "results/tables/test_predictions.csv"
)

df["Date"] = pd.to_datetime(
    df["Date"],
    utc=True
)

model_names = [
    "Elastic Net",
    "Decision Tree",
    "XGBoost",
    "LightGBM"
]

# Experimental assumption:
# 10 basis points = 0.1% transaction cost
TRANSACTION_COST = 0.001

TRADING_DAYS = 252


# --------------------------------------------------
# 2. Helper function for performance metrics
# --------------------------------------------------

def calculate_metrics(
    returns,
    directional_accuracy=np.nan,
    position_changes=0
):
    returns = pd.Series(returns).fillna(0)

    cumulative = (1 + returns).cumprod()

    total_return = cumulative.iloc[-1] - 1

    n_days = len(returns)

    annualized_return = (
        cumulative.iloc[-1] **
        (TRADING_DAYS / n_days)
        - 1
    )

    annualized_volatility = (
        returns.std() * np.sqrt(TRADING_DAYS)
    )

    if returns.std() != 0:
        sharpe = (
            np.sqrt(TRADING_DAYS)
            * returns.mean()
            / returns.std()
        )
    else:
        sharpe = np.nan

    running_max = cumulative.cummax()

    drawdown = (
        cumulative / running_max - 1
    )

    max_drawdown = drawdown.min()

    return {
        "Total_Return": total_return,
        "Annualized_Return": annualized_return,
        "Annualized_Volatility": annualized_volatility,
        "Sharpe_Ratio": sharpe,
        "Max_Drawdown": max_drawdown,
        "Directional_Accuracy": directional_accuracy,
        "Position_Changes": position_changes
    }


# --------------------------------------------------
# 3. Buy-and-hold benchmark
# --------------------------------------------------

df["Buy_and_Hold_Return"] = df["Actual_Return"]

df["Buy_and_Hold_Cumulative"] = (
    1 + df["Buy_and_Hold_Return"]
).cumprod()

buy_hold_metrics = calculate_metrics(
    df["Buy_and_Hold_Return"]
)

results = [{
    "Strategy": "Buy and Hold",
    **buy_hold_metrics
}]


# --------------------------------------------------
# 4. Model trading strategies
# --------------------------------------------------

for model in model_names:

    # Long if predicted return >= 0
    # Short if predicted return < 0
    position_col = f"{model}_Position"

    df[position_col] = np.where(
        df[model] >= 0,
        1,
        -1
    )

    # Gross strategy return before costs
    gross_return_col = (
        f"{model}_Gross_Return"
    )

    df[gross_return_col] = (
        df[position_col]
        * df["Actual_Return"]
    )

    # --------------------------------------------------
    # Turnover
    # --------------------------------------------------
    #
    # First day:
    # Assume we establish the initial position.
    #
    # Later days:
    # abs(position_t - position_t-1)
    #
    # Long -> Short gives turnover = 2
    # Short -> Long gives turnover = 2
    # Long -> Long gives turnover = 0
    #

    turnover_col = f"{model}_Turnover"

    df[turnover_col] = (
        df[position_col]
        .diff()
        .abs()
        .fillna(
            abs(df[position_col].iloc[0])
        )
    )

    # Number of times direction changed
    position_changes = (
        df[position_col]
        .diff()
        .fillna(0)
        .ne(0)
        .sum()
    )

    # Transaction cost
    cost_col = f"{model}_Transaction_Cost"

    df[cost_col] = (
        df[turnover_col]
        * TRANSACTION_COST
    )

    # Net return after transaction costs
    net_return_col = (
        f"{model}_Net_Return"
    )

    df[net_return_col] = (
        df[gross_return_col]
        - df[cost_col]
    )

    # Cumulative net return
    cumulative_col = (
        f"{model}_Cumulative_Net"
    )

    df[cumulative_col] = (
        1 + df[net_return_col]
    ).cumprod()

    # Directional accuracy
    directional_accuracy = (
        np.sign(df[model])
        == np.sign(df["Actual_Return"])
    ).mean()

    metrics = calculate_metrics(
        df[net_return_col],
        directional_accuracy,
        position_changes
    )

    results.append({
        "Strategy": model,
        **metrics
    })


# --------------------------------------------------
# 5. Create results table
# --------------------------------------------------

results_df = pd.DataFrame(results)


# --------------------------------------------------
# 6. Save files
# --------------------------------------------------

output_dir = Path("results/tables")

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    output_dir / "backtest_results_detailed.csv",
    index=False
)

df.to_csv(
    output_dir / "backtest_daily_detailed.csv",
    index=False
)


# --------------------------------------------------
# 7. Print results
# --------------------------------------------------

print("\n==============================================")
print("DETAILED BACKTEST RESULTS")
print("==============================================")

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print(
    "\nTransaction cost assumption:",
    f"{TRANSACTION_COST * 100:.2f}% per unit turnover"
)

print(
    "\nNumber of test observations:",
    len(df)
)


# --------------------------------------------------
# 8. Position summary
# --------------------------------------------------

print("\nPOSITION SUMMARY")

for model in model_names:

    position_col = f"{model}_Position"

    long_days = (
        df[position_col] == 1
    ).sum()

    short_days = (
        df[position_col] == -1
    ).sum()

    changes = (
        df[position_col]
        .diff()
        .fillna(0)
        .ne(0)
        .sum()
    )

    print(
        f"{model}: "
        f"Long={long_days}, "
        f"Short={short_days}, "
        f"Direction changes={changes}"
    )