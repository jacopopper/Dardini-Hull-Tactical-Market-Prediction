# Hull Tactical Market Prediction

> **Kaggle Competition**: <https://www.kaggle.com/competitions/hull-tactical-market-prediction>
>
> This project predicts S&P 500 daily excess returns and converts them into portfolio weights in `[0, 2]`. The objective is to maximize the competition's adjusted Sharpe ratio, which penalizes excess volatility and sustained underperformance relative to the market index.
>
> **Key results**: adjusted Sharpe **0.8605** on the final temporal holdout (1,500 days, 200-day purge gap), compared with the baseline (`weight = 1.0`) score of **0.7151**.

## Objective

The project predicts daily S&P 500 excess returns and converts them into tactical portfolio weights. The strategy uses only `submission/data/train.csv`; Kaggle `test.csv` is excluded because it has no target and is intended only for leaderboard submissions.

## Results

| Validation Step          | Adjusted Sharpe |
| ------------------------ | --------------: |
| Initial untuned mean CV  |          0.7246 |
| Best inner CV mean       |      **0.8526** |

| Configuration         | Adjusted Sharpe | Annualized Return | Volatility Ratio | Active Days |
| --------------------- | --------------: | ----------------: | ---------------: | ----------: |
| Model strategy        |      **0.8605** |         **20.2%** |            1.020 |       29.5% |
| Baseline weight = 1.0 |          0.7151 |             16.8% |            1.000 |          -- |

The final evaluation uses a chronological split: a tuning block, a 200-day purge gap, and a final holdout of 1,500 trading days. Optuna uses only the inner walk-forward CV folds built from the tuning block; the final holdout is evaluated once after parameter selection.

## Interpretation

The best inner CV score shows how the selected strategy parameters performed across several historical validation windows before the final holdout evaluation. The model's holdout adjusted Sharpe of 0.8605 is above the baseline score of 0.7151, so the adaptive weighting adds value in this split.

The fold diagnostics are not uniform across time. Some validation windows benefit more from adaptive weights than others, while the mean fold weights remain close to 1.0 in most folds. This indicates that the strategy usually stays near market exposure and changes exposure moderately.

## Deliverables

```text
.
├── doc/
│   ├── presentation.tex
│   └── presentation.pdf
├── submission/
│   ├── readme.md
│   ├── main.ipynb
│   ├── main.pdf
│   └── data/
│       └── train.csv
└── readme.md
```

## Startup Notes

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

Install the required packages:

```bash
pip install numpy pandas scikit-learn xgboost hmmlearn optuna jupyter matplotlib
```

Run the project:

```bash
cd submission
jupyter notebook main.ipynb
```

Run all notebook cells from top to bottom.

## References

- Kaggle competition: <https://www.kaggle.com/competitions/hull-tactical-market-prediction>
- Moreira, A. & Muir, T. (2017). _Volatility-Managed Portfolios_. Journal of Finance, 72(4), 1611--1644.
- Chen, T. & Guestrin, C. (2016). _XGBoost: A Scalable Tree Boosting System_. KDD 2016.
