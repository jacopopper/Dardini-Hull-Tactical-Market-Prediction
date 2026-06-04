# Hull Tactical Market Prediction

https://www.kaggle.com/competitions/hull-tactical-market-prediction

## Objective

The project predicts daily S&P 500 excess returns and converts them into portfolio weights in `[0, 2]`. The goal is to maximize the competition adjusted Sharpe ratio, which penalizes both excessive volatility and sustained underperformance versus the market.

## Results

| Configuration         | Adjusted Sharpe | Annualized Return | Volatility Ratio | Active Days |
| --------------------- | --------------: | ----------------: | ---------------: | ----------: |
| Model strategy        |      **0.8605** |         **20.2%** |            1.020 |       29.5% |
| Baseline weight = 1.0 |          0.7151 |             16.8% |            1.000 |          -- |

The final evaluation uses only `train.csv`, split chronologically into a tuning block and a final holdout of 1,500 trading days with a 200-row purge gap.

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
