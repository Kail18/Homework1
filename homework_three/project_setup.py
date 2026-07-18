"""Utilities for creating the Homework 3 project structure."""

from pathlib import Path
from typing import Final


PROJECT_DIRECTORY_NAME: Final[str] = "homework_three"

DIRECTORIES: Final[tuple[str, ...]] = (
    "data/raw",
    "data/splits",
    "src",
    "outputs/baseline",
    "outputs/tuning",
    "outputs/optimized",
    "outputs/plots",
)

SOURCE_FILES: Final[tuple[str, ...]] = (
    "src/__init__.py",
    "src/dataset.py",
    "src/model.py",
    "src/train.py",
    "src/evaluate.py",
    "src/tune.py",
    "src/utils.py",
)


def _starter_content(file_path: Path) -> str:
    """Return minimal starter content for a newly created Python module."""
    if file_path.name == "__init__.py":
        return '"""Homework 3 source package."""\n'

    module_name = file_path.stem.replace("_", " ").title()
    return f'"""{module_name} utilities for Homework 3."""\n'


def create_homework_three_structure(
    repository_root: str | Path | None = None,
) -> Path:
    """Create the Homework 3 folders and starter source files.

    The function is idempotent: it may be called repeatedly without deleting or
    overwriting existing directories and source files.

    Args:
        repository_root: Homework 3 project directory. When omitted, the
            directory containing this file is used.

    Returns:
        The absolute path to the Homework 3 project directory.
    """

    if repository_root is None:
        project_root = Path(__file__).resolve().parent
    else:
        project_root = Path(repository_root).expanduser().resolve()

    for relative_directory in DIRECTORIES:
        (project_root / relative_directory).mkdir(
            parents=True,
            exist_ok=True,
        )

    for relative_file in SOURCE_FILES:
        file_path = project_root / relative_file
        file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not file_path.exists():
            file_path.write_text(
                _starter_content(file_path),
                encoding="utf-8",
            )

    return project_root
