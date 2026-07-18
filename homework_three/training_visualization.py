from pathlib import Path

import matplotlib.pyplot as plt


def plot_training_history(
    history,
    output_directory: str,
    model_name: str,
) -> None:
    """Plot training and validation accuracy and loss."""

    output_path = Path(
        output_directory
    )

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_name = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    epochs = range(
        1,
        len(
            history.history[
                "accuracy"
            ]
        ) + 1,
    )

    # =============================================
    # Accuracy Curve
    # =============================================

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        epochs,
        history.history[
            "accuracy"
        ],
        label="Training Accuracy",
    )

    plt.plot(
        epochs,
        history.history[
            "val_accuracy"
        ],
        label="Validation Accuracy",
    )

    plt.title(
        f"{model_name} Training and Validation Accuracy"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.legend()

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.savefig(
        output_path
        / f"{file_name}_accuracy_curve.png"
    )

    plt.close()

    # =============================================
    # Loss Curve
    # =============================================

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        epochs,
        history.history[
            "loss"
        ],
        label="Training Loss",
    )

    plt.plot(
        epochs,
        history.history[
            "val_loss"
        ],
        label="Validation Loss",
    )

    plt.title(
        f"{model_name} Training and Validation Loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.legend()

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.savefig(
        output_path
        / f"{file_name}_loss_curve.png"
    )

    plt.close()


def plot_model_comparison(
    baseline_history,
    optimized_history,
    optimized_confusion_matrix_path: str,
    output_directory: str,
) -> None:
    """
    Create a combined visualization comparing baseline and
    optimized CNN training curves with the optimized model's
    confusion matrix.
    """

    output_path = Path(
        output_directory
    )

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    confusion_matrix_path = Path(
        optimized_confusion_matrix_path
    )

    if not confusion_matrix_path.exists():

        raise FileNotFoundError(
            "Optimized confusion matrix does not exist: "
            f"{confusion_matrix_path}"
        )

    # =============================================
    # Create Figure Layout
    # =============================================

    figure = plt.figure(
        figsize=(18, 10)
    )

    grid = figure.add_gridspec(
        2,
        3,
        width_ratios=[
            1,
            1,
            1.2,
        ],
    )

    # =============================================
    # Baseline Accuracy
    # =============================================

    baseline_accuracy_axis = (
        figure.add_subplot(
            grid[
                0,
                0,
            ]
        )
    )

    baseline_epochs = range(
        1,
        len(
            baseline_history.history[
                "accuracy"
            ]
        ) + 1,
    )

    baseline_accuracy_axis.plot(
        baseline_epochs,
        baseline_history.history[
            "accuracy"
        ],
        label="Training Accuracy",
    )

    baseline_accuracy_axis.plot(
        baseline_epochs,
        baseline_history.history[
            "val_accuracy"
        ],
        label="Validation Accuracy",
    )

    baseline_accuracy_axis.set_title(
        "Baseline CNN Accuracy"
    )

    baseline_accuracy_axis.set_xlabel(
        "Epoch"
    )

    baseline_accuracy_axis.set_ylabel(
        "Accuracy"
    )

    baseline_accuracy_axis.legend()

    baseline_accuracy_axis.grid(
        True
    )

    # =============================================
    # Baseline Loss
    # =============================================

    baseline_loss_axis = (
        figure.add_subplot(
            grid[
                1,
                0,
            ]
        )
    )

    baseline_loss_axis.plot(
        baseline_epochs,
        baseline_history.history[
            "loss"
        ],
        label="Training Loss",
    )

    baseline_loss_axis.plot(
        baseline_epochs,
        baseline_history.history[
            "val_loss"
        ],
        label="Validation Loss",
    )

    baseline_loss_axis.set_title(
        "Baseline CNN Loss"
    )

    baseline_loss_axis.set_xlabel(
        "Epoch"
    )

    baseline_loss_axis.set_ylabel(
        "Loss"
    )

    baseline_loss_axis.legend()

    baseline_loss_axis.grid(
        True
    )

    # =============================================
    # Optimized Accuracy
    # =============================================

    optimized_accuracy_axis = (
        figure.add_subplot(
            grid[
                0,
                1,
            ]
        )
    )

    optimized_epochs = range(
        1,
        len(
            optimized_history.history[
                "accuracy"
            ]
        ) + 1,
    )

    optimized_accuracy_axis.plot(
        optimized_epochs,
        optimized_history.history[
            "accuracy"
        ],
        label="Training Accuracy",
    )

    optimized_accuracy_axis.plot(
        optimized_epochs,
        optimized_history.history[
            "val_accuracy"
        ],
        label="Validation Accuracy",
    )

    optimized_accuracy_axis.set_title(
        "Optimized CNN Accuracy"
    )

    optimized_accuracy_axis.set_xlabel(
        "Epoch"
    )

    optimized_accuracy_axis.set_ylabel(
        "Accuracy"
    )

    optimized_accuracy_axis.legend()

    optimized_accuracy_axis.grid(
        True
    )

    # =============================================
    # Optimized Loss
    # =============================================

    optimized_loss_axis = (
        figure.add_subplot(
            grid[
                1,
                1,
            ]
        )
    )

    optimized_loss_axis.plot(
        optimized_epochs,
        optimized_history.history[
            "loss"
        ],
        label="Training Loss",
    )

    optimized_loss_axis.plot(
        optimized_epochs,
        optimized_history.history[
            "val_loss"
        ],
        label="Validation Loss",
    )

    optimized_loss_axis.set_title(
        "Optimized CNN Loss"
    )

    optimized_loss_axis.set_xlabel(
        "Epoch"
    )

    optimized_loss_axis.set_ylabel(
        "Loss"
    )

    optimized_loss_axis.legend()

    optimized_loss_axis.grid(
        True
    )

    # =============================================
    # Optimized Confusion Matrix
    # =============================================

    confusion_matrix_axis = (
        figure.add_subplot(
            grid[
                :,
                2,
            ]
        )
    )

    confusion_matrix_image = (
        plt.imread(
            confusion_matrix_path
        )
    )

    confusion_matrix_axis.imshow(
        confusion_matrix_image
    )

    confusion_matrix_axis.set_title(
        "Optimized CNN Confusion Matrix"
    )

    confusion_matrix_axis.axis(
        "off"
    )

    # =============================================
    # Save Comparison Figure
    # =============================================

    figure.suptitle(
        "Baseline vs. Optimized CNN Evaluation",
        fontsize=16,
    )

    plt.tight_layout()

    comparison_path = (
        output_path
        / "training_comparison.png"
    )

    plt.savefig(
        comparison_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "\nModel comparison plot saved to: "
        f"{comparison_path}"
    )