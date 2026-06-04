# Hull Tactical Market Prediction — Submission Package

This folder contains the exam software and data package for the Kaggle challenge [Hull Tactical Market Prediction](https://www.kaggle.com/competitions/hull-tactical-market-prediction).

## Project Description

The project builds a machine learning pipeline for tactical asset allocation on the S&P 500. It predicts daily excess returns, estimates market regimes, and converts model confidence into portfolio weights between `0` and `2`.

The competition metric is an adjusted Sharpe ratio. It penalizes excessive volatility and underperformance relative to the market, so the strategy defaults to full market exposure (`weight = 1.0`) unless the model has enough confidence to take an active position.

## Data

The `data/` folder contains:

- `train.csv`: Kaggle training data with targets, risk-free rates, returns, and anonymized market features.

Only `train.csv` is used. The Kaggle `test.csv` file is excluded because it has no target and is intended only for leaderboard submissions.

## Results

| Configuration         | Adjusted Sharpe | Annualized Return | Volatility Ratio | Active Days |
| --------------------- | --------------: | ----------------: | ---------------: | ----------: |
| Model strategy        |      **0.8605** |         **20.2%** |            1.020 |       29.5% |
| Baseline weight = 1.0 |          0.7151 |             16.8% |            1.000 |          -- |

Evaluation uses a strict chronological split: the last 1,500 trading days are held out, with a 200-row purge gap between tuning and evaluation data.

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
