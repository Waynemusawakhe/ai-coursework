"""K-means clustering of synthetic fitness-tracker data (CMPG313 Artificial Intelligence, 2025).

Generates three groups of users (active, moderate, inactive) by daily steps and nightly sleep,
then clusters them with K-means. Compared with the original assignment, this version:
  - scales both features before clustering (steps run into thousands, sleep into single digits,
    so unscaled K-means is driven almost entirely by steps),
  - checks the choice of k with the elbow method and silhouette scores,
  - saves the charts to files instead of only showing them.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(15)


def generate_users(n, steps, sleep):
    return np.column_stack((rng.integers(*steps, n), rng.uniform(*sleep, n)))


groups = {
    "active": generate_users(rng.integers(100, 1000), (8000, 12000), (7.5, 9.0)),
    "moderate": generate_users(rng.integers(100, 1000), (5000, 8000), (6.5, 8.0)),
    "inactive": generate_users(rng.integers(100, 1000), (2000, 5000), (6.0, 7.5)),
}
data = np.vstack(list(groups.values()))
scaled = StandardScaler().fit_transform(data)

# Choosing k: inertia (elbow) and silhouette score for k = 2..7
ks = range(2, 8)
inertia, silhouette = [], []
for k in ks:
    model = KMeans(n_clusters=k, random_state=42, n_init=10).fit(scaled)
    inertia.append(model.inertia_)
    silhouette.append(silhouette_score(scaled, model.labels_))
best_k = ks[int(np.argmax(silhouette))]
print("Silhouette by k:", {k: round(s, 3) for k, s in zip(ks, silhouette)})

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
a1.plot(list(ks), inertia, marker="o"); a1.set_title("Elbow method"); a1.set_xlabel("k"); a1.set_ylabel("Inertia")
a2.plot(list(ks), silhouette, marker="o"); a2.set_title("Silhouette score"); a2.set_xlabel("k")
fig.tight_layout(); fig.savefig(OUT / "choosing_k.png", dpi=150); plt.close(fig)

# Final model with k = 3 (the number of groups generated)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(scaled)
labels = kmeans.labels_

fig, ax = plt.subplots(figsize=(7, 5))
sc = ax.scatter(data[:, 0], data[:, 1], c=labels, cmap="viridis", alpha=0.5, s=12)
ax.set_xlabel("Steps per day"); ax.set_ylabel("Hours slept per night")
ax.set_title("K-means clusters (scaled features, k = 3)")
fig.tight_layout(); fig.savefig(OUT / "clusters.png", dpi=150); plt.close(fig)

for i in np.argsort([data[labels == c, 0].mean() for c in range(3)])[::-1]:
    pts = data[labels == i]
    print(f"Cluster: {len(pts):4d} users, mean steps {pts[:, 0].mean():7.0f}, mean sleep {pts[:, 1].mean():.2f} h")
print(f"Best k by silhouette: {best_k}")
