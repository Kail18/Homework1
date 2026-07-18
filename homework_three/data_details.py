from pathlib import Path
from collections import Counter


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