from breezeml import datasets, fit, predict

print("Running regression test...")
df = datasets.diabetes()
model = fit(df, "target")
yhat = predict(model, df.drop(columns=["target"]))
print("First 10 predictions:", yhat[:10])
