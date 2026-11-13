import tensorflow as tf

# BUILD #
model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(
            shape=(28, 28, 1),
        ),
        tf.keras.layers.Conv2D(
            filters=8,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
        ),
        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2),
        ),
        tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
        ),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(
            units=10,
            activation="softmax",
        ),
    ]
)

# COMPLIE #
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# SUMMARY #
print(
    "Output shapes: (28,28,1) -> (28,28,8) -> (14,14,8) -> (14,14,16) -> (3136,) -> (10,)"
)
model.summary()
