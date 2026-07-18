from pathlib import Path

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
BATCH_SIZE = 32
RANDOM_SEED = 42


def load_split_data(split_csv: str) -> tuple:
    """Load the saved dataset split CSV and separate each dataset split."""

    csv_path = Path(split_csv)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset split CSV does not exist: {csv_path}"
        )

    split_data = pd.read_csv(csv_path)

    train_data = split_data[
        split_data["split"] == "train"
    ].copy()

    validation_data = split_data[
        split_data["split"] == "validation"
    ].copy()

    test_data = split_data[
        split_data["split"] == "test"
    ].copy()

    return train_data, validation_data, test_data

def create_class_mapping(
    split_data: pd.DataFrame,
) -> dict:
    """Create a consistent numeric mapping for each fish class."""

    class_names = sorted(
        split_data["class_name"].unique()
    )

    return {
        class_name: index
        for index, class_name in enumerate(class_names)
    }

def preprocess_image(
    image_path: tf.Tensor,
    label: tf.Tensor,
) -> tuple:
    """Load, convert, resize, and normalize a single image."""

    image = tf.io.read_file(image_path)

    image = tf.io.decode_image(
        image,
        # This is force the RGBA files int RGB format which is 3 channels
        channels=3,
        expand_animations=False,
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        [IMAGE_HEIGHT, IMAGE_WIDTH],
    )

    image = tf.cast(
        image,
        tf.float32,
    )

    image = image / 255.0

    return image, label

def create_augmentation_pipeline() -> tf.keras.Sequential:
    """Create data augmentation layers for training images."""

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip(
                "horizontal",
                seed=RANDOM_SEED,
            ),
            tf.keras.layers.RandomRotation(
                0.05,
                seed=RANDOM_SEED,
            ),
            tf.keras.layers.RandomContrast(
                0.20,
                seed=RANDOM_SEED,
            ),
        ],
        name="data_augmentation",
    )

def create_dataset(
    data: pd.DataFrame,
    class_mapping: dict,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = False,
) -> tf.data.Dataset:
    """Create a TensorFlow dataset from image paths and class labels."""

    image_paths = data["image_path"].values

    labels = data["class_name"].map(
        class_mapping
    ).values

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            image_paths,
            labels,
        )
    )

    dataset = dataset.map(
        preprocess_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(data),
            seed=RANDOM_SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.batch(
        batch_size
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset

def create_datasets(
    split_csv: str,
    batch_size: int = BATCH_SIZE,
) -> tuple:
    """Create training, validation, and testing TensorFlow datasets."""

    train_data, validation_data, test_data = load_split_data(
        split_csv
    )

    all_data = pd.concat(
        [
            train_data,
            validation_data,
            test_data,
        ],
        ignore_index=True,
    )

    class_mapping = create_class_mapping(
        all_data
    )

    train_dataset = create_dataset(
        train_data,
        class_mapping,
        batch_size=batch_size,
        shuffle=True,
    )

    validation_dataset = create_dataset(
        validation_data,
        class_mapping,
        batch_size=batch_size,
        shuffle=False,
    )

    test_dataset = create_dataset(
        test_data,
        class_mapping,
        batch_size=batch_size,
        shuffle=False,
    )

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
        class_mapping,
    )

def visualize_augmentations(
    train_dataset: tf.data.Dataset,
    class_mapping: dict,
    number_of_images: int = 4,
) -> None:
    """Display original preprocessed images and augmented versions."""

    augmentation_pipeline = create_augmentation_pipeline()

    # Reverse the class mapping so numeric labels can be
    # converted back into readable fish class names.
    reverse_class_mapping = {
        class_id: class_name
        for class_name, class_id in class_mapping.items()
    }

    # Take one batch from the training dataset.
    for images, labels in train_dataset.take(1):

        number_of_images = min(
            number_of_images,
            images.shape[0],
        )

        figure, axes = plt.subplots(
            number_of_images,
            4,
            figsize=(12, 3 * number_of_images),
        )

        for row in range(number_of_images):
            image = images[row]
            label = int(labels[row].numpy())

            class_name = reverse_class_mapping[label]

            # Column 1: Preprocessed image
            axes[row, 0].imshow(image)
            axes[row, 0].set_title(
                f"Original\n{class_name}"
            )
            axes[row, 0].axis("off")

            # Columns 2-4: Different random augmentations
            for column in range(1, 4):

                augmented_image = augmentation_pipeline(
                    tf.expand_dims(image, axis=0),
                    training=True,
                )

                clipped_image = tf.clip_by_value(
                    augmented_image[0],
                    0.0,
                    1.0,
                )

                axes[row, column].imshow(
                    clipped_image
                )

                axes[row, column].set_title(
                    f"Augmented {column}"
                )

                axes[row, column].axis("off")

        plt.tight_layout()
        plt.show()