import numpy as np

x = np.array([1.0, 2.0])
W1 = np.array(
    [
        [0.2, 0.4],
        [0.6, 0.1],
        [0.3, 0.5],
    ]
)
b1 = np.array([0.1, -0.1, 0.2])
W2 = np.array(
    [
        [0.5, 0.2, 0.4],
        [0.1, 0.7, 0.3],
    ]
)
b2 = np.array([0.1, 0.2])

relu = lambda z: np.maximum(0, z)
z1 = W1 @ x + b1
a1 = relu(z1)

z2 = W2 @ a1 + b2
shifted = z2 - np.max(z2)
a2 = np.exp(shifted) / np.exp(shifted).sum()

print(z1, a1, z2, a2)
print(np.isclose(a2.sum(), 1.0))
