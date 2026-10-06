from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# Setup
# --------------------------------------------------

input_file = "results/tables/backtest_daily_detailed.csv"

output_dir = Path("results/figures")
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_file)

df["Date"] = pd.to_datetime(df["Date"])


# --------------------------------------------------
# GRAPH 1: Cumulative strategy performance
# --------------------------------------------------

plt.figure(figsize=(11, 6))

plt.plot(
    df["Date"],
    df["Buy_and_Hold_Cumulative"],
    label="Buy and Hold"
)

for model in [
    "Elastic Net",
    "Decision Tree",
    "XGBoost",
    "LightGBM"
]:

    plt.plot(
        df["Date"],
        df[f"{model}_Cumulative_Net"],
        label=model
    )

plt.title(
    "Cumulative Trading Performance on Test Period"
)

plt.xlabel("Date")
plt.ylabel("Growth of €1")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    output_dir / "cumulative_returns.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# GRAPH 2: Final total returns
# --------------------------------------------------

results = pd.read_csv(
    "results/tables/backtest_results_detailed.csv"
)

plt.figure(figsize=(9, 5))

plt.bar(
    results["Strategy"],
    results["Total_Return"] * 100
)

plt.title(
    "Total Return by Trading Strategy"
)

plt.xlabel("Strategy")
plt.ylabel("Total Return (%)")

plt.xticks(
    rotation=20,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    output_dir / "total_returns.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# GRAPH 3: Sharpe ratios
# --------------------------------------------------

plt.figure(figsize=(9, 5))

plt.bar(
    results["Strategy"],
    results["Sharpe_Ratio"]
)

plt.title(
    "Risk-Adjusted Performance"
)

plt.xlabel("Strategy")
plt.ylabel("Sharpe Ratio")

plt.xticks(
    rotation=20,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    output_dir / "sharpe_ratios.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# GRAPH 4: Maximum drawdown
# --------------------------------------------------

plt.figure(figsize=(9, 5))

plt.bar(
    results["Strategy"],
    results["Max_Drawdown"] * 100
)

plt.title(
    "Maximum Drawdown by Strategy"
)

plt.xlabel("Strategy")
plt.ylabel("Maximum Drawdown (%)")

plt.xticks(
    rotation=20,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    output_dir / "max_drawdown.png",
    dpi=300
)

plt.close()


print("\nVisualizations created successfully.")

print("\nSaved files:")
print("results/figures/cumulative_returns.png")
print("results/figures/total_returns.png")
print("results/figures/sharpe_ratios.png")
print("results/figures/max_drawdown.png")