import numpy as np
from sklearn.cluster import KMeans

image = np.ones((120, 160, 3))

pixels = image.reshape(-1, 3)

model = KMeans(
    n_clusters=16,
    random_state=42,
    n_init=10,
)

labels = model.fit_predict(pixels)

centers = model.cluster_centers_

reconstructed = centers[labels].reshape(image.shape)

print(reconstructed.shape)
