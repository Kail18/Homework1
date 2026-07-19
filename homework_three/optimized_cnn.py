import tensorflow as tf

from image_preprocessing import (
    create_augmentation_pipeline,
)


def build_optimized_cnn(
    number_of_classes: int,
    dropout_rate: float = 0.5,
) -> tf.keras.Model:
    """Build an optimized CNN with configurable Dropout."""

    data_augmentation = (
        create_augmentation_pipeline()
    )

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(128, 128, 3),
            ),

            data_augmentation,

            tf.keras.layers.Conv2D(
                32,
                (3, 3),
                activation="relu",
                padding="same",
                name="conv_block_1",
            ),

            tf.keras.layers.MaxPooling2D(
                (2, 2),
                name="pool_1",
            ),

            tf.keras.layers.Conv2D(
                64,
                (3, 3),
                activation="relu",
                padding="same",
                name="conv_block_2",
            ),

            tf.keras.layers.MaxPooling2D(
                (2, 2),
                name="pool_2",
            ),

            tf.keras.layers.Conv2D(
                128,
                (3, 3),
                activation="relu",
                padding="same",
                name="conv_block_3",
            ),

            tf.keras.layers.MaxPooling2D(
                (2, 2),
                name="pool_3",
            ),

            tf.keras.layers.GlobalAveragePooling2D(
                name="global_average_pooling",
            ),

            tf.keras.layers.Dropout(
                dropout_rate,
                name="dropout_1",
            ),

            tf.keras.layers.Dense(
                64,
                activation="relu",
                name="dense_classifier",
            ),

            tf.keras.layers.Dropout(
                dropout_rate,
                name="dropout_2",
            ),

            tf.keras.layers.Dense(
                number_of_classes,
                activation="softmax",
                name="classification_output",
            ),
        ],
        name="optimized_fish_cnn",
    )

    return model


def compile_optimized_cnn(
    model: tf.keras.Model,
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    """Compile the optimized CNN with a configurable learning rate."""

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
    )

    model.compile(
        optimizer=optimizer,
        loss=(
            "sparse_categorical_crossentropy"
        ),
        metrics=[
            "accuracy",
        ],
    )

    return model