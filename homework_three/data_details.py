from pathlib import Path
from collections import Counter
from PIL import Image


def count_file_types(data_directory: str) -> dict:
    """Count the file extensions found in the dataset directory."""

    directory = Path(data_directory)

    if not directory.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {directory}"
        )

    file_counts = Counter()

    for file_path in directory.rglob("*"):
        if file_path.is_file():
            file_counts[file_path.suffix.lower()] += 1

    return dict(file_counts)

def count_classes(data_directory: str) -> dict:
    """Count the number of files in each class directory."""

    directory = Path(data_directory)

    if not directory.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {directory}"
        )

    class_counts = {}

    for class_directory in directory.iterdir():
        if class_directory.is_dir():

            image_count = sum(
                1
                for file_path in class_directory.rglob("*")
                if file_path.is_file()
            )

            class_counts[class_directory.name] = image_count

    return class_counts

def inspect_image_properties(data_directory: str) -> dict:
    """Inspect image dimensions, color modes, and corrupted image files."""

    directory = Path(data_directory)

    if not directory.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {directory}"
        )

    supported_extensions = {".jpg", ".jpeg", ".png"}

    dimension_counts = Counter()
    color_mode_counts = Counter()
    corrupted_images = []

    for file_path in directory.rglob("*"):
        if (
            file_path.is_file()
            and file_path.suffix.lower() in supported_extensions
        ):
            try:
                with Image.open(file_path) as image:
                    image.verify()

                # Reopen after verify() because verify() checks the file
                # without fully loading the image data.
                with Image.open(file_path) as image:
                    width, height = image.size
                    dimension_counts[(width, height)] += 1
                    color_mode_counts[image.mode] += 1

            except (OSError, ValueError) as error:
                corrupted_images.append(
                    {
                        "file": str(file_path),
                        "error": str(error),
                    }
                )

    return {
        "dimensions": dict(dimension_counts),
        "color_modes": dict(color_mode_counts),
        "corrupted_images": corrupted_images,
    }

