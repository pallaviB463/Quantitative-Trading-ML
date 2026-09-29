# Quantitative Trading with Machine Learning

## Project Overview

This project investigates whether machine learning models can predict Volkswagen (VW) stock returns using historical stock-return features and related financial data.

We compare Elastic Net, Decision Tree, XGBoost and LightGBM against a zero-return baseline. The project will also evaluate whether model predictions can be used in a trading strategy.

## Dataset

- Source: Dataset from the reference research project.
- Original period: January 2005 to December 2019.
- Features: Historical return-based features.
- Prediction targets: Conventional forward returns over 1, 5 and 20 trading days.

The original future-return columns use a reverse-return formula. We calculate conventional forward returns directly from closing prices and remove rows without the required future prices.

The raw and processed datasets are excluded from GitHub. Each teammate must obtain the dataset separately and place it in `data/raw/`.

## Completed Work

- Project environment and folder structure.
- Dataset inspection and preprocessing.
- Forward-return target correction.
- Chronological training, validation and test split.
- A 20-observation gap between the splits.
- Training-only feature selection and preprocessing.
- Zero-return baseline.
- Initial training of Elastic Net, Decision Tree, XGBoost and LightGBM.
- Elastic Net hyperparameter tuning.
- Decision Tree hyperparameter tuning.
- Validation results saved as CSV files.

## Current Results

All results below are from the 1-day return prediction task on the validation set.

| Model | Validation RMSE |
|---|---:|
| Zero-return baseline | 0.022496 |
| Original Elastic Net | 0.027063 |
| Tuned Elastic Net | 0.022576 |
| Original Decision Tree | 0.023150 |
| Tuned Decision Tree | 0.022540 |
| Original XGBoost | 0.023221 |
| Original LightGBM | 0.023879 |

The tuned Elastic Net has zero non-zero feature coefficients and predicts a constant return. The tuned Decision Tree improves on the original configuration.

No model has yet beaten the zero-return baseline on validation RMSE.

## Remaining Work

- XGBoost hyperparameter tuning.
- LightGBM hyperparameter tuning.
- Final model comparison.
- Trading strategy and backtesting.
- Performance visualizations.
- Evaluation of the untouched test set.
- Experiments for the other forecast horizons.
- Final report, presentation and live demo.

## Collaboration

Contributors should create a separate Git branch, commit their changes and open a pull request for review before merging into `main`.
