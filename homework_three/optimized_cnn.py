import tensorflow as tf
from tensorflow.keras import layers, models


def build_optimized_cnn(number_of_classes, image_height=128, image_width=128):
    """
    Build optimized CNN architecture.

    Improvements over baseline:
    - Replace Flatten with GlobalAveragePooling2D
    - Reduce dense layer size
    - Add dropout regularization
    - Reduce total parameter count
    """

    model = models.Sequential(
    [
        layers.Input(
            shape=(
                image_height,
                image_width,
                3,
            )
        )
    ],
    name="optimized_fish_cnn",
)

    # -------------------------------------------------
    # Data Augmentation
    # -------------------------------------------------

    model.add(
        layers.RandomFlip(
            "horizontal",
            name="random_flip"
        )
    )

    model.add(
        layers.RandomRotation(
            0.05,
            name="random_rotation"
        )
    )

    model.add(
        layers.RandomContrast(
            0.2,
            name="random_contrast"
        )
    )


    # -------------------------------------------------
    # Convolution Block 1
    # Input: 128 x 128 x 3
    # -------------------------------------------------

    model.add(
        layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
            name="conv_block_1"
        )
    )

    model.add(
        layers.MaxPooling2D(
            pool_size=(2, 2),
            name="pool_1"
        )
    )


    # -------------------------------------------------
    # Convolution Block 2
    # -------------------------------------------------

    model.add(
        layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
            name="conv_block_2"
        )
    )

    model.add(
        layers.MaxPooling2D(
            pool_size=(2, 2),
            name="pool_2"
        )
    )


    # -------------------------------------------------
    # Convolution Block 3
    # -------------------------------------------------

    model.add(
        layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
            name="conv_block_3"
        )
    )

    model.add(
        layers.MaxPooling2D(
            pool_size=(2, 2),
            name="pool_3"
        )
    )


    # -------------------------------------------------
    # Classification Head
    # -------------------------------------------------

    # Replacement for Flatten()
    # Reduces:
    # 16 x 16 x 128 = 32,768 values
    # to:
    # 128 feature averages
    model.add(
        layers.GlobalAveragePooling2D(
            name="global_average_pooling"
        )
    )


    # Regularization
    model.add(
        layers.Dropout(
            0.5,
            name="dropout_1"
        )
    )


    # Smaller dense classifier
    model.add(
        layers.Dense(
            units=64,
            activation="relu",
            name="dense_classifier"
        )
    )


    model.add(
        layers.Dropout(
            0.3,
            name="dropout_2"
        )
    )


    # Output layer
    model.add(
        layers.Dense(
            units=number_of_classes,
            activation="softmax",
            name="classification_output"
        )
    )


    return model



def compile_optimized_cnn(
    model,
    learning_rate=0.001
):
    """
    Compile optimized CNN.

    Initial experiment keeps:
    - Adam optimizer
    - Learning rate = 0.001
    - Sparse categorical crossentropy

    These match the baseline for fair comparison.
    """

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy"
        ]
    )

    return model



def print_model_summary(model):
    """
    Print optimized model architecture.
    """

    model.summary()