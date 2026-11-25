import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(784,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2,
)

# probabilities = model.predict(X_train[:10])
probabilities = model.predict(
    X_test[:10]
)  # test the model on the test set, not the training set.
# predictions = tf.argmax(probabilities, axis=0)
predictions = tf.argmax(
    probabilities, axis=1
)  # index on the feature axis to output the most possible category.
