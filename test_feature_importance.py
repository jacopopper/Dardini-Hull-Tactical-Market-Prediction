import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load data
original_train_df = pd.read_csv("data/train.csv")

# We want to check feature importances for predicting the forward_returns
# Let's drop rows without forward_returns
df = original_train_df.dropna(subset=['forward_returns']).copy()

# Fill NAs in features with 0 for this quick test
features = [col for col in df.columns if col not in ['date_id', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns']]
X = df[features].fillna(0)
y = df['forward_returns']

print(f"Training Random Forest on {len(features)} features (including {sum(c.startswith('D') for c in features)} 'D' columns)...")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X, y)

# Get feature importances
importances = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\n--- Top 20 Most Important Features ---")
print(importances.head(20))

# Check specifically for 'D' columns
d_cols = [c for c in features if c.startswith('D')]
print(f"\n--- Importance of 'D' (Dummy) columns ---")
d_importances = importances[importances['Feature'].isin(d_cols)]
print(d_importances.head(10)) 
print(f"... and {len(d_cols) - 10} more.")

print(f"\nTotal importance of all 'D' columns combined: {d_importances['Importance'].sum():.4f}")
