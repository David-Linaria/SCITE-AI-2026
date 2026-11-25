import numpy as np


def relu(z):
    return np.maximum(0, z)


def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)


x = np.array([1.0, 2.0])

W1 = np.array(
    [
        [0.2, 0.4],
        [0.5, 0.1],
        [0.3, 0.6],
    ]
)
b1 = np.array([0.1, 0.2, 0.3])

W2 = np.array(
    [
        [0.4, 0.2, 0.5],
        [0.1, 0.6, 0.3],
    ]
)
b2 = np.array([0.1, 0.2])

z1 = W1 @ x + b1
a1 = relu(z1)

z2 = W2 @ a1 + b2
output = softmax(z2)

print(output)
