from breezeml import datasets, fit, predict, creator

print("Easter Egg:", creator())
df = datasets.iris()
model = fit(df, "species")
yhat = predict(model, df.drop(columns=["species"]))
print("First 10 predictions:", yhat[:10])
