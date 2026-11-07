import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

X = np.array(
    [
        [1, 1],
        [1, 2],
        [2, 1],
        [8, 8],
        [8, 9],
        [9, 8],
    ]
)

model = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10,
)

labels = model.fit_predict(X)

print("Cluster Labels: ", labels)
print("Cluster Centroids: ", model.cluster_centers_)
print("Inertia: ", model.inertia_)

fig, ax = plt.subplots(figsize=(10, 8))

ax.scatter(
    X[:, 0],
    X[:, 1],
    c=labels,
    cmap="viridis",
)

ax.scatter(
    model.cluster_centers_[:, 0],
    model.cluster_centers_[:, 1],
    marker="X",
    s=180,
    c="red",
    label="Centroids",
)

ax.set_title("K-Means Cluster")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.legend()

plt.show()