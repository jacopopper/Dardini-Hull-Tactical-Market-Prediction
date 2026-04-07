import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import sharpe

# Load data
original_train_df = pd.read_csv("data/train.csv")
vix_df = pd.read_csv("VIX_History.csv")["CLOSE"]

# Prepare features and target
train_cols = original_train_df.columns
vol_cols = [c for c in train_cols if "V" in c]

vol_df = original_train_df[vol_cols].copy()
vol_df["target"] = vix_df
vol_df = vol_df.dropna()

X = vol_df.drop('target', axis=1)
y = vol_df['target']

# Train Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Get predictions back to original dataframe space
new_df = original_train_df.copy()
new_df = new_df.drop(vol_cols, axis=1)
new_df.loc[X.index, "VIX"] = model.predict(X)

# We need the forward_returns and risk_free_rate columns from the original training data 
# to calculate the Sharpe Ratio using the score function
# We should drop rows where we didn't have VIX predictions (due to dropna earlier)
eval_df = new_df.loc[X.index].copy()

# Strategy logic: 
# If Predicted VIX is high (e.g., above the 75th percentile of the training data VIX), overweight (e.g., 1.5)
# Else, neutral weight (1.0)
vix_75th = eval_df["VIX"].quantile(0.75)
eval_df["prediction"] = np.where(eval_df["VIX"] > vix_75th, 1.5, 1.0)

# The 'score' function expects DataFrames for 'solution' and 'submission'.
# 'solution' needs 'risk_free_rate' and 'forward_returns'
# 'submission' needs 'prediction'
solution_df = eval_df[['risk_free_rate', 'forward_returns']].copy()
submission_df = eval_df[['prediction']].copy()

# Calculate score
try:
    final_score = sharpe.score(solution=solution_df, submission=submission_df, row_id_column_name="date_id") # Note: score function definition in sharpe.py doesn't actually use row_id_column_name
    print(f"Custom Sharpe Ratio Score: {final_score:.6f}")
except Exception as e:
    print(f"Error calculating score: {e}")
