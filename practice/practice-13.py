import numpy as np

Q = np.array(
    [
        [1.0, 0.0],
        [0.0, 1.0],
    ]
)

K = np.array(
    [
        [1.0, 0.0],
        [0.0, 1.0],
    ]
)

V = np.array(
    [
        [2.0, 1.0],
        [1.0, 3.0],
    ]
)

d_k = K.shape[-1]

scores = (Q @ K.T) / np.sqrt(d_k)

shifted = scores - np.max(scores, axis=-1, keepdims=True)

exp_scores = np.exp(shifted)

weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

output = weights @ V

print(weights.shape)
print(output.shape)
print(np.sum(weights, axis=-1))
