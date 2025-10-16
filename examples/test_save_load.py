from breezeml import datasets, fit, save, load, predict

print("Running save/load test...")
df = datasets.iris()
model = fit(df, "species")
save(model, "iris_model.joblib")
loaded = load("iris_model.joblib")
X = df.drop(columns=["species"])
print("Match first 10? ", (predict(model, X)[:10] == predict(loaded, X)[:10]).all())
