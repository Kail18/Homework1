import tensorflow as tf

from image_preprocessing import create_augmentation_pipeline


def build_baseline_cnn(
    number_of_classes: int,
    image_height: int = 128,
    image_width: int = 128,
) -> tf.keras.Model:
    """Build the baseline CNN for fish image classification."""

    data_augmentation = create_augmentation_pipeline()

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(
                    image_height,
                    image_width,
                    3,
                )
            ),

            # Apply augmentation only during training.
            data_augmentation,

            # Convolution block 1
            tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=(3, 3),
                activation="relu",
                padding="same",
            ),

            tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2),
            ),

            # Convolution block 2
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=(3, 3),
                activation="relu",
                padding="same",
            ),

            tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2),
            ),

            # Convolution block 3
            tf.keras.layers.Conv2D(
                filters=128,
                kernel_size=(3, 3),
                activation="relu",
                padding="same",
            ),

            tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2),
            ),

            # Convert feature maps into a single vector.
            tf.keras.layers.Flatten(),

            # Fully connected classification layer.
            tf.keras.layers.Dense(
                256,
                activation="relu",
            ),

            # Final multiclass output layer.
            tf.keras.layers.Dense(
                number_of_classes,
                activation="softmax",
            ),
        ],
        name="baseline_fish_cnn",
    )

    return model

def compile_baseline_cnn(
    model: tf.keras.Model,
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    """Compile the baseline CNN."""

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model

def load_saved_model(model_path: str):
    """Load a saved Keras model."""

    model = tf.keras.models.load_model(
        model_path
    )

    model.summary()

    return model