import numpy as np

X_train = np.array(
    [
        [0, 0],
        [1, 0],
        [0, 1],
        [4, 4],
        [5, 4],
        [4, 5],
    ]
)

y_train = np.array([0, 0, 0, 1, 1, 1])

query = np.array([3, 3])
k = 3

difference = X_train - query
distance = np.sqrt(
    np.sum(
        difference**2,
        axis=1,
    )
)

indicies = np.argsort(distance)[:k]
label = y_train[indicies]
label_sorted = np.bincount(label)
final_prediction = np.argmax(label_sorted)

print(distance)
print(indicies)
print(label)
print(final_prediction)
