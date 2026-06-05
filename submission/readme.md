# Hull Tactical Market Prediction — Submission Package

> **Kaggle Competition**: [Hull Tactical Market Prediction](https://www.kaggle.com/competitions/hull-tactical-market-prediction)
>
> This folder contains the exam software and data package for a complete tactical market prediction workflow. The notebook predicts S&P 500 daily excess returns, estimates regimes, and converts model confidence into portfolio weights in `[0, 2]`.
>
> **Key results**: adjusted Sharpe **0.8605** on the final temporal holdout (1,500 days, 200-day purge gap), compared with the baseline (`weight = 1.0`) score of **0.7151**.

## Project Description

The competition metric is an adjusted Sharpe ratio. It penalizes excess volatility and sustained underperformance relative to the market index, so the strategy defaults to full market exposure (`weight = 1.0`) unless the model has enough confidence to take an active position.

## Data

The `data/` folder contains:

- `train.csv`: Kaggle training data with targets, risk-free rates, returns, and anonymized market features.

Only `train.csv` is used. The Kaggle `test.csv` file is excluded because it has no target and is intended only for leaderboard submissions.

## Results

| Validation Step          | Adjusted Sharpe |
| ------------------------ | --------------: |
| Initial untuned mean CV  |          0.7246 |
| Best inner CV mean       |      **0.8526** |

| Configuration         | Adjusted Sharpe | Annualized Return | Volatility Ratio | Active Days |
| --------------------- | --------------: | ----------------: | ---------------: | ----------: |
| Model strategy        |      **0.8605** |         **20.2%** |            1.020 |       29.5% |
| Baseline weight = 1.0 |          0.7151 |             16.8% |            1.000 |          -- |

Evaluation uses a strict chronological split: a tuning block, a 200-day purge gap, and a final holdout of 1,500 trading days. Optuna uses only the inner walk-forward CV folds built from the tuning block; the final holdout is evaluated once after parameter selection.

## Interpretation

The best inner CV score shows how the selected strategy parameters performed across several historical validation windows before the final holdout evaluation. The final holdout score is the notebook's held-out offline estimate.

The model's holdout adjusted Sharpe of 0.8605 is above the baseline weight-1 score of 0.7151, so the adaptive weighting adds value in this split without relying on the excluded Kaggle test file. The fold diagnostics show that performance varies across time: some validation windows benefit more from adaptive weights than others. Mean fold weights stay close to 1.0 in most folds, indicating moderate exposure changes.

## Files

```text
submission/
├── readme.md
├── main.ipynb
├── main.pdf
└── data/
    └── train.csv
```

## Installation and Startup

Prerequisites:

- Python 3.9+
- `numpy`
- `pandas`
- `scikit-learn`
- `xgboost`
- `hmmlearn`
- `optuna`
- `jupyter`
- `matplotlib`

Install dependencies:

```bash
pip install numpy pandas scikit-learn xgboost hmmlearn optuna jupyter matplotlib
```

Start the notebook:

```bash
jupyter notebook main.ipynb
```

Run all cells sequentially. The notebook is self-contained and includes the competition scoring function, feature engineering, model training, walk-forward validation, Optuna tuning, and final holdout reporting.

## Notebook Workflow

1. Load libraries, plotting style, and fixed ex-ante constants.
2. Define the competition score and utility functions.
3. Load `data/train.csv` and create the strict temporal split.
4. Build engineered features with train-only fit and leakage-safe transform.
5. Inspect engineered features with tables and diagnostic plots.
6. Fit the regime detector, return model, and position-sizing logic.
7. Build the inner walk-forward validation cache.
8. Run Optuna strategy tuning on cached validation folds.
9. Refit the pipeline on the full tuning block and evaluate the untouched holdout.
10. Report selected parameters, validation scores, holdout scores, and plots.

## References

- Kaggle competition: <https://www.kaggle.com/competitions/hull-tactical-market-prediction>
- Moreira, A. & Muir, T. (2017). _Volatility-Managed Portfolios_. Journal of Finance, 72(4), 1611--1644.
- Chen, T. & Guestrin, C. (2016). _XGBoost: A Scalable Tree Boosting System_. KDD 2016.
