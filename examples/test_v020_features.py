"""
BreezeML v0.2.0 — Test all new classification features.
Run: python examples/test_v020_features.py
"""
import sys
sys.path.insert(0, ".")

from breezeml import classifiers, datasets

df = datasets.iris()
target = "species"

print("=" * 60)
print("🧪 BreezeML v0.2.0 — Classification Feature Tests")
print("=" * 60)

# ── Test 1: New classifiers ──────────────────────────────────
print("\n📌 Test 1: New classifiers")
for name, func in [
    ("KNN", classifiers.knn),
    ("Gradient Boosting", classifiers.gradient_boosting),
    ("AdaBoost", classifiers.adaboost),
    ("Extra Trees", classifiers.extra_trees),
    ("MLP", classifiers.mlp),
]:
    model, report = func(df, target)
    print(f"  ✅ {name}: accuracy={report['accuracy']}, f1={report['f1']}")

# ── Test 2: Compare leaderboard ──────────────────────────────
print("\n📌 Test 2: Classifier Comparison")
results = classifiers.compare(df, target)
print(f"  ✅ Compared {len(results)} classifiers. Best: {results[0]['classifier']}")

# ── Test 3: Detailed report ──────────────────────────────────
print("\n📌 Test 3: Detailed Report")
info = classifiers.detailed_report(df, target)
print(f"  ✅ Accuracy:  {info['accuracy']}")
print(f"  ✅ Precision: {info['precision']}")
print(f"  ✅ Recall:    {info['recall']}")
print(f"  ✅ ROC-AUC:   {info['roc_auc']}")
print(f"  ✅ Confusion Matrix: {info['confusion_matrix']}")

# ── Test 4: Quick tune ───────────────────────────────────────
print("\n📌 Test 4: Quick Tune")
model, params, report = classifiers.quick_tune(df, target, algo="knn", n_iter=10)
print(f"  ✅ Best KNN params: {params}")
print(f"  ✅ Tuned report: {report}")

print("\n" + "=" * 60)
print("🎉 All v0.2.0 tests passed! 🔥✨")
print("=" * 60)
