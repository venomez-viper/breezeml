from breezeml import datasets, fit, save, load, predict

print("Running save/load test...")

# Train a simple model on Iris
df = datasets.iris()
model = fit(df, "species")

# Save to disk
model_path = "iris_model.joblib"
save(model, model_path)
print(f"Model saved to: {model_path}")

# Load back
loaded = load(model_path)
print("Model loaded.")

# Sanity check: predictions should match (on same data)
X = df.drop(columns=["species"])
orig_pred = predict(model, X)[:10]
loaded_pred = predict(loaded, X)[:10]

print("Original preds (first 10):", orig_pred)
print("Loaded   preds (first 10):", loaded_pred)

# Optional: assert equality for the first 50 rows
same = (orig_pred[:50] == loaded_pred[:50]).all()
print("Do first 50 predictions match?", bool(same))
