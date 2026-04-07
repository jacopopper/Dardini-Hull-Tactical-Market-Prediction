import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

original_train_df = pd.read_csv("data/train.csv")

vix_df = pd.read_csv("VIX_History.csv")["CLOSE"]
#print(vix_df.tail)

train_cols = original_train_df.columns
vol_cols = [c for c in train_cols if "V" in c]
#print(vol_cols)
vol_df = original_train_df[vol_cols].copy()
vol_df["target"] = vix_df
vol_df = vol_df.dropna()

# Prepare features (X) and target (y)
X = vol_df.drop('target', axis=1)
y = vol_df['target']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model (using Random Forest for better performance)
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared: {r2:.4f}")

# Plot actual vs predicted VIX
plt.figure(figsize=(12, 5))

# Subplot 1: Scatter plot
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
line_coords = np.array([y_test.min(), y_test.max()])
plt.plot(line_coords, line_coords, 'r--', lw=2, label="Perfect Fit")
plt.xlabel('Actual VIX')
plt.ylabel('Predicted VIX')
plt.title('Actual vs Predicted VIX')
plt.legend()

# Subplot 2: Time-series overlay for the last 500 test samples (sorted by original index)
plt.subplot(1, 2, 2)
# Align actuals and predictions to properly sort them backwards in time
results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).sort_index()

# Plot the last 500 available points in test set
last_500_idx = results_df.index[-500:]
plt.plot(last_500_idx, results_df['Actual'].loc[last_500_idx], label='Actual VIX', marker='o', markersize=4)
plt.plot(last_500_idx, results_df['Predicted'].loc[last_500_idx], label='Predicted VIX', marker='x', markersize=4)
plt.xlabel('Original Index (Time Proxy)')
plt.ylabel('VIX Value')
plt.title('VIX Tracking (Last 500 Test Samples)')
plt.legend()

plt.tight_layout()
plt.show()


new_df = original_train_df.copy()
new_df = new_df.drop(vol_cols, axis=1)
new_df.loc[X.index, "VIX"] = model.predict(X)


# --- FEATURE IMPORTANCE FOR NEW_DF ---
print("\n--- Feature Importance Analysis ---")
# Keep all columns except VIX and date_id (if it exists)
features = [c for c in new_df.columns if c not in ['VIX', 'date_id']]

X_all = new_df[features].dropna()
y_all = new_df.loc[X_all.index, 'VIX']

print(f"Training Random Forest on {len(features)} features to predict VIX...")
rf_all = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_all.fit(X_all, y_all)

# Get feature importances
importances = pd.DataFrame({
    'Feature': features,
    'Importance': rf_all.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nTop 20 Most Important Features:")
print(importances.head(20))

# Check specifically for 'D' columns
d_cols = [c for c in features if c.startswith('D')]
print(f"\nImportance of 'D' (Dummy) columns (out of {len(d_cols)} total 'D' columns):")
d_importances = importances[importances['Feature'].isin(d_cols)]
print(d_importances)

print(f"\nTotal importance of all 'D' columns combined: {d_importances['Importance'].sum():.4f}")

