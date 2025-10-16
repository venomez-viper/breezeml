"""
🌬️ BreezeML Classifiers Demo 🔥✨
Demonstrates Logistic Regression, Naive Bayes, Decision Tree, and Random Forest.
"""

from breezeml import datasets, classifiers

df = datasets.iris()
print("=== BreezeML Classifiers Demo ===\n")

model, report = classifiers.logistic(df, "species")
print("Logistic Regression:", report)

model, report = classifiers.gaussian_nb(df, "species")
print("Gaussian Naive Bayes:", report)

model, report = classifiers.decision_tree(df, "species")
print("Decision Tree:", report)

model, report = classifiers.random_forest(df, "species")
print("Random Forest:", report)

print("\n✅ Classifier tests completed successfully!\n")
