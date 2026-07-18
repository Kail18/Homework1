import matplotlib.pyplot as plt
from pathlib import Path


def plot_training_history(
    history,
    output_directory: str,
    model_name: str,
) -> None:
    """
    Plot and save training/validation accuracy and loss curves.

    Args:
        history:
            Keras History object returned by model.fit().

        output_directory:
            Directory where plots will be saved.

        model_name:
            Name of model for plot titles and filenames.
    """

    output_path = Path(output_directory)
    output_path.mkdir(
        parents=True,
        exist_ok=True
    )

    epochs = range(
        1,
        len(history.history["accuracy"]) + 1
    )

    # -------------------------------
    # Accuracy Curve
    # -------------------------------

    plt.figure(figsize=(8, 6))

    plt.plot(
        epochs,
        history.history["accuracy"],
        label="Training Accuracy",
    )

    plt.plot(
        epochs,
        history.history["val_accuracy"],
        label="Validation Accuracy",
    )

    plt.title(
        f"{model_name} Training and Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()

    plt.savefig(
        output_path /
        f"{model_name.lower()}_accuracy_curve.png",
        bbox_inches="tight",
        dpi=300,
    )

    plt.close()


    # -------------------------------
    # Loss Curve
    # -------------------------------

    plt.figure(figsize=(8, 6))

    plt.plot(
        epochs,
        history.history["loss"],
        label="Training Loss",
    )

    plt.plot(
        epochs,
        history.history["val_loss"],
        label="Validation Loss",
    )

    plt.title(
        f"{model_name} Training and Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()

    plt.savefig(
        output_path /
        f"{model_name.lower()}_loss_curve.png",
        bbox_inches="tight",
        dpi=300,
    )

    plt.close()