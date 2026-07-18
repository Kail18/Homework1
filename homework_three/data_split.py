from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def create_stratified_split(
    data_directory: str,
    output_directory: str,
    train_size: float = 0.70,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create stratified training, validation, and testing splits.

    Each top-level directory in the dataset is treated as a class.

    Args:
        data_directory: Path to the fish dataset.
        output_directory: Directory where the split CSV will be saved.
        train_size: Proportion of images assigned to training.
        validation_size: Proportion assigned to validation.
        test_size: Proportion assigned to testing.
        random_state: Seed used to make the split reproducible.

    Returns:
        A DataFrame containing image paths, class labels, and split assignments.
    """

    if round(train_size + validation_size + test_size, 10) != 1.0:
        raise ValueError(
            "Train, validation, and test sizes must add up to 1.0."
        )

    dataset_path = Path(data_directory)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {dataset_path}"
        )

    image_paths = []
    class_labels = []

    # Each folder directly inside Fish is treated as one fish class.
    for class_directory in dataset_path.iterdir():
        if not class_directory.is_dir():
            continue

        for image_path in class_directory.rglob("*"):
            if (
                image_path.is_file()
                and image_path.suffix.lower() in SUPPORTED_EXTENSIONS
            ):
                image_paths.append(str(image_path))
                class_labels.append(class_directory.name)

    if not image_paths:
        raise ValueError(
            f"No supported images were found in {dataset_path}"
        )

    # First split:
    # 70% training
    # 30% temporary set for validation + testing
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths,
        class_labels,
        train_size=train_size,
        stratify=class_labels,
        random_state=random_state,
    )

    # The temporary set contains 30% of the original dataset.
    # Divide it evenly into 15% validation and 15% testing.
    validation_ratio = validation_size / (
        validation_size + test_size
    )

    (
        validation_paths,
        test_paths,
        validation_labels,
        test_labels,
    ) = train_test_split(
        temp_paths,
        temp_labels,
        train_size=validation_ratio,
        stratify=temp_labels,
        random_state=random_state,
    )

    train_data = pd.DataFrame(
        {
            "image_path": train_paths,
            "class_name": train_labels,
            "split": "train",
        }
    )

    validation_data = pd.DataFrame(
        {
            "image_path": validation_paths,
            "class_name": validation_labels,
            "split": "validation",
        }
    )

    test_data = pd.DataFrame(
        {
            "image_path": test_paths,
            "class_name": test_labels,
            "split": "test",
        }
    )

    split_data = pd.concat(
        [
            train_data,
            validation_data,
            test_data,
        ],
        ignore_index=True,
    )

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / "dataset_splits.csv"

    split_data.to_csv(
        csv_path,
        index=False,
    )

    return split_data

def display_split_summary(split_data: pd.DataFrame) -> None:
    """Print the number of images from each class in each dataset split."""

    summary = pd.crosstab(
        split_data["class_name"],
        split_data["split"],
    )

    summary["total"] = summary.sum(axis=1)

    print("\nDataset Split Summary")
    print("---------------------")
    print(summary)

    print("\nTotal Images Per Split")
    print("----------------------")
    print(split_data["split"].value_counts())