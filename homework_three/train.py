from pathlib import Path

import tensorflow as tf


def train_baseline_model(
    model: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    validation_dataset: tf.data.Dataset,
    output_directory: str,
    epochs: int = 30,
) -> tf.keras.callbacks.History:
    """Train the baseline CNN using early stopping and model checkpointing."""

    output_path = Path(output_directory)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        output_path / "best_baseline_model.keras"
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[
            early_stopping,
            model_checkpoint,
        ],
    )

    return history