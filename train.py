"""
Standalone training script exported by BreezeML.

This file reproduces the exact pipeline BreezeML trained, using only
scikit-learn / pandas / joblib. No breezeml import required.

Usage:
    1. Point DATA_PATH at your training CSV.
    2. python train.py
"""

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC

DATA_PATH = "YOUR_DATA.csv"
TARGET = "species"
RANDOM_STATE = 42
TEST_SIZE = 0.2

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=[TARGET])
y = df[TARGET]

numeric = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
categorical = []

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical),
])

model = SVC(kernel='linear', probability=True)
pipeline = Pipeline([("pre", preprocessor), ("model", model)])

stratify = y if (y.nunique() > 1 and y.nunique() < len(y)) else None
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify)
pipeline.fit(X_train, y_train)

preds = pipeline.predict(X_test)
print("accuracy:", round(accuracy_score(y_test, preds), 4))
print("f1 (weighted):", round(f1_score(y_test, preds, average="weighted"), 4))

joblib.dump(pipeline, "model.joblib")
print("Saved trained pipeline to model.joblib")
