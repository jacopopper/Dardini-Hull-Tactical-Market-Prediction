import json

with open("main.ipynb", "r") as f:
    nb = json.load(f)

# Find the target cell. It should be the last one, or we just append a new one
target_idx = -1
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code" and "df = new_df.dropna(subset=['forward_returns']).copy()" in "".join(cell["source"]):
        target_idx = i
        break

new_source = """df = new_df.dropna(subset=['forward_returns']).copy()

# Fill NAs in features with 0 for this quick test
features = [col for col in df.columns if col not in ['date_id', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns']]
X = df[features].fillna(0)
y = df['forward_returns']

print(f"Training Random Forest on {len(features)} features")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X, y)

importances = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

display(importances)

#Drop features following the 1/N heuristic
n_features = len(features)
threshold = 1.0 / n_features
print(f"\\n1/N Threshold: {threshold:.4f}")

# Select features that meet the threshold
important_features = importances[importances['Importance'] >= threshold]['Feature'].tolist()
dropped_features = importances[importances['Importance'] < threshold]['Feature'].tolist()

print(f"Keeping {len(important_features)} features.")
print(f"Dropping {len(dropped_features)} features.")

# Update the DataFrame by dropping the unimportant features
new_df = new_df.drop(columns=dropped_features)
print(f"New DataFrame shape after dropping features: {new_df.shape}")
"""

# Format source for Jupyter (list of strings with newlines)
source_lines = [line + "\n" if i < len(new_source.split('\n')) - 1 else line for i, line in enumerate(new_source.split('\n'))]

if target_idx != -1:
    nb["cells"][target_idx]["source"] = source_lines
else:
    # If the cell doesn't exist yet, append it
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines
    })

with open("main.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
