import numpy as np

X = np.array(
    [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]
)

K = np.array(
    [
        [1, 0],
        [0, -1],
    ]
)


def conv2d_valid(X, K):
    # a convolution layer with padding=0, stride=1.
    X_new = np.ones(
        (
            X.shape[0] - K.shape[0] + 1,
            X.shape[1] - K.shape[1] + 1,
        )
    )

    for i in range(X.shape[0] - K.shape[0] + 1):
        for j in range(X.shape[1] - K.shape[1] + 1):
            X_sub = X[i : i + K.shape[0], j : j + K.shape[1]]
            X_new[i][j] = (X_sub * K).sum()

    return X_new


X_new = conv2d_valid(X, K)
print(X_new, X_new.shape)
