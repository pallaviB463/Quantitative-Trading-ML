# Quantitative Trading with Machine Learning

## Overview

This project investigates whether machine learning models can predict future Volkswagen (VW) stock returns using historical return information from VW and companies in its supply chain.

The project is inspired by the work **"Quantitative Trading with Machine Learning" by René M. Glawion**.

Four regression models are evaluated:

- Elastic Net
- Decision Tree Regressor
- XGBoost
- LightGBM

The models are compared with a zero-return baseline and evaluated across 1-day, 5-day, and 20-day forecast horizons.

A trading experiment is also performed using model predictions and compared with a buy-and-hold strategy.

---

## Project Objectives

The main objectives are:

1. Prepare financial time-series data for machine learning.
2. Predict future VW stock returns.
3. Compare linear and tree-based regression models.
4. Evaluate models using RMSE and MAE.
5. Study prediction performance across different forecast horizons.
6. Convert return predictions into trading signals.
7. Compare ML-based strategies with buy-and-hold.
8. Analyze trading performance using return and risk metrics.

---

## Dataset

The dataset is derived from the replication material associated with the reference project.

It contains:

- Volkswagen stock information
- Historical VW returns
- Historical returns of companies in the VW supply chain
- Macroeconomic and financial variables

The original dataset contains:

- 3,809 observations
- 1,292 columns

For this project, historical return-based features are selected.

After preprocessing:

- 868 candidate return features were identified.
- 756 features were available in the training period.
- Forecast targets were created for 1, 5, and 20 trading days.

The raw and processed datasets are excluded from GitHub.

---

## Target Construction

During dataset inspection, the supplied future-return columns were found to use the reverse-return formulation.

To obtain conventional forward returns, targets were calculated directly from VW closing prices:

```text
Target_Return_h = Close(t+h) / Close(t) - 1
```

where `h` represents the forecast horizon.

Targets were generated for:

- 1 day
- 5 days
- 20 days

Rows without observable future prices were removed.

---

## Data Splitting

Because this is financial time-series data, random train-test splitting is not used.

The observations are sorted chronologically and divided into training, validation, and test periods.

For the main 1-day experiment:

- Training samples: 2,632
- Validation samples: 548
- Test samples: 569

Temporal gaps are used between partitions to reduce the risk of future-return labels crossing dataset boundaries.

The test set remains unseen during model selection and hyperparameter tuning.

---

## Models

### Elastic Net

Elastic Net combines L1 and L2 regularization.

The selected 1-day configuration was:

```text
alpha = 0.1
l1_ratio = 0.9
```

Strong regularization reduced all feature coefficients to zero during the validation experiment, causing the model to behave approximately as a constant-return predictor.

### Decision Tree

Selected configuration:

```text
max_depth = 2
min_samples_leaf = 10
```

A shallow tree generalized better than more complex trees during validation.

### XGBoost

Selected configuration:

```text
n_estimators = 200
max_depth = 2
learning_rate = 0.01
```

### LightGBM

Selected configuration:

```text
n_estimators = 200
max_depth = 4
num_leaves = 16
learning_rate = 0.01
```

---

## 1-Day Validation Results

| Model | Validation RMSE |
|---|---:|
| Zero-return baseline | 0.022496 |
| Decision Tree | 0.022540 |
| Elastic Net | 0.022576 |
| XGBoost | 0.022643 |
| LightGBM | 0.022898 |

The zero-return baseline remained highly competitive during validation.

---

## Final 1-Day Test Results

| Model | RMSE | MAE |
|---|---:|---:|
| Elastic Net | **0.016701** | 0.012852 |
| Zero-return baseline | 0.016706 | **0.012808** |
| XGBoost | 0.016851 | 0.012935 |
| LightGBM | 0.016886 | 0.013094 |
| Decision Tree | 0.018934 | 0.013181 |

Elastic Net achieved the lowest test RMSE, although its improvement over the zero-return baseline was extremely small.

The zero-return baseline achieved the lowest MAE.

---

## Multi-Horizon Results

Fixed model configurations were evaluated across 1-day, 5-day, and 20-day forecast horizons.

| Model | 1-Day RMSE | 5-Day RMSE | 20-Day RMSE |
|---|---:|---:|---:|
| Elastic Net | **0.016700** | 0.037245 | **0.065780** |
| Decision Tree | 0.018934 | **0.037188** | 0.068469 |
| XGBoost | 0.016864 | 0.037317 | 0.066290 |
| LightGBM | 0.016856 | 0.037907 | 0.066325 |
| Zero-return baseline | 0.016706 | 0.037304 | 0.066111 |

Prediction error generally increases as the forecast horizon becomes longer.

Elastic Net produced the lowest RMSE at the 1-day and 20-day horizons, while Decision Tree produced the lowest RMSE at the 5-day horizon.

The 5-day and 20-day experiments reuse configurations selected during the 1-day experiment and therefore should be interpreted as fixed-configuration comparisons rather than independently optimized models.

---

## Trading Strategy

For the primary 1-day backtest, model predictions are converted into trading positions:

```text
Predicted return >= 0  -> Long
Predicted return < 0   -> Short
```

The strategy is compared with buy-and-hold.

An experimental transaction cost of 0.1% per unit of turnover is also included.

---

## Detailed 1-Day Backtest

| Strategy | Total Return | Annualized Return | Sharpe Ratio | Maximum Drawdown |
|---|---:|---:|---:|---:|
| Buy and Hold | **42.22%** | **16.88%** | **0.7206** | -29.13% |
| Elastic Net | 42.08% | 16.83% | 0.7190 | -29.13% |
| Decision Tree | 42.08% | 16.83% | 0.7190 | -29.13% |
| XGBoost | 34.29% | 13.95% | 0.6246 | -31.23% |
| LightGBM | 8.94% | 3.86% | 0.2753 | -47.12% |

Elastic Net and Decision Tree remained long throughout the test period. Therefore, their strategies effectively reproduced buy-and-hold before transaction costs.

XGBoost changed direction only a few times, while LightGBM traded substantially more frequently.

---

## Threshold Experiment

Trading thresholds of:

- 0%
- 0.25%
- 0.5%
- 1%

were evaluated.

A position is opened only when the magnitude of the predicted return exceeds the selected threshold.

The threshold experiment helps examine whether filtering low-confidence return predictions changes trading behavior.

For 5-day and 20-day horizons, consecutive target periods overlap. Therefore, directly compounding every predicted return would overstate realistic portfolio performance.

For this reason, longer-horizon cumulative returns are treated as exploratory results rather than realistic trading profits.

---

## Visualizations

Generated figures are stored in:

```text
results/figures/
```

The project generates:

- `cumulative_returns.png`
- `total_returns.png`
- `sharpe_ratios.png`
- `max_drawdown.png`

---

## Project Structure

```text
Quantitative-Trading-ML/
|
|-- data/
|   |-- raw/
|   `-- processed/
|
|-- models/
|
|-- notebooks/
|
|-- reports/
|
|-- results/
|   |-- figures/
|   `-- tables/
|
|-- src/
|   |-- preprocessing.py
|   |-- train_models.py
|   |-- tune_elasticnet.py
|   |-- tune_decision_tree.py
|   |-- tune_xgboost.py
|   |-- tune_lightgbm.py
|   |-- final_evaluation.py
|   |-- verify_backtest_timing.py
|   |-- backtest.py
|   |-- create_visualizations.py
|   |-- multihorizon_evaluation.py
|   `-- threshold_backtest.py
|
|-- .gitignore
|-- README.md
`-- requirements.txt
```

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd Quantitative-Trading-ML
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

### 1. Preprocess the dataset

```bash
python src/preprocessing.py
```

### 2. Train the initial models

```bash
python src/train_models.py
```

### 3. Run model tuning

```bash
python src/tune_elasticnet.py
python src/tune_decision_tree.py
python src/tune_xgboost.py
python src/tune_lightgbm.py
```

### 4. Run final test evaluation

```bash
python src/final_evaluation.py
```

### 5. Verify target/backtest timing

```bash
python src/verify_backtest_timing.py
```

### 6. Run the detailed backtest

```bash
python src/backtest.py
```

### 7. Run multi-horizon evaluation

```bash
python src/multihorizon_evaluation.py
```

### 8. Run threshold experiments

```bash
python src/threshold_backtest.py
```

### 9. Generate visualizations

```bash
python src/create_visualizations.py
```

---

## Key Findings

1. Stock-return prediction is difficult, and a zero-return baseline is highly competitive.

2. Increasing model complexity did not automatically improve forecasting performance.

3. Elastic Net achieved the lowest 1-day test RMSE, but its advantage over the baseline was extremely small.

4. Prediction RMSE increased with longer forecast horizons.

5. The 1-day ML trading strategies did not outperform buy-and-hold in this test period.

6. Elastic Net and Decision Tree effectively behaved as always-long strategies during the 1-day test period.

7. Transaction costs particularly affected strategies with more frequent position changes.

8. Longer-horizon directional accuracy improved in the exploratory threshold experiment, but overlapping returns prevent interpreting naive cumulative returns as realistic portfolio profits.

---

## Limitations

- The project uses a simplified chronological split rather than reproducing the complete rolling-window evaluation procedure of the reference research.
- Hyperparameters for the 5-day and 20-day experiments were not independently optimized.
- The primary trading strategy uses closing-price return assumptions.
- A fixed transaction-cost assumption is used rather than historical bid-ask spreads and brokerage costs.
- Longer-horizon returns overlap and require a more sophisticated portfolio simulation for realistic trading-profit estimates.
- Results from one historical period may not generalize to future market conditions.

---

## Reference

René M. Glawion, *Quantitative Trading with Machine Learning*, Stanford University CS229 project.

The original study evaluates Elastic Net, Decision Trees, XGBoost, and LightGBM for predicting Volkswagen stock returns using supply-chain information.

---

## Conclusion

This project demonstrates an end-to-end machine learning workflow for quantitative return prediction.

Although the machine learning models produced competitive forecasting errors, they did not provide a clear economic advantage over simple benchmarks in the 1-day trading experiment.

The results highlight the importance of chronological evaluation, strong baselines, careful target construction, transaction costs, and realistic backtesting when applying machine learning to financial markets.
