"""
🌬️ BreezeML Clustering Demo 🔥✨
Demonstrates KMeans, Agglomerative, and DBSCAN clustering.
"""

from breezeml import datasets, clustering

df = datasets.wine()
X = df.drop(columns=["class"])

print("=== BreezeML Clustering Demo ===\n")

res = clustering.kmeans(X, n_clusters=3)
print("KMeans silhouette:", res["silhouette"])

res = clustering.agglomerative(X, n_clusters=3)
print("Agglomerative silhouette:", res["silhouette"])

res = clustering.dbscan(X, eps=1.0, min_samples=5)
print("DBSCAN clusters:", len(set(res["labels"])))

print("\n✅ Clustering tests completed successfully!\n")
