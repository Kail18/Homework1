import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.metrics import confusion_matrix


BASELINE_HISTORY = (
    "homework_three/outputs/baseline/training_history.json"
)

OPTIMIZED_HISTORY = (
    "homework_three/outputs/optimized/training_history.json"
)


OUTPUT_DIRECTORY = Path(
    "homework_three/outputs/model_comparison"
)


OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)


def load_history(path):

    with open(path, "r") as file:
        return json.load(file)



def plot_training_comparison():

    baseline = load_history(
        BASELINE_HISTORY
    )

    optimized = load_history(
        OPTIMIZED_HISTORY
    )


    figure, axes = plt.subplots(
        2,
        2,
        figsize=(14,10)
    )


    # Baseline Accuracy

    axes[0,0].plot(
        baseline["accuracy"],
        label="Training"
    )

    axes[0,0].plot(
        baseline["val_accuracy"],
        label="Validation"
    )

    axes[0,0].set_title(
        "Baseline CNN Accuracy"
    )

    axes[0,0].set_xlabel(
        "Epoch"
    )

    axes[0,0].set_ylabel(
        "Accuracy"
    )

    axes[0,0].legend()



    # Baseline Loss

    axes[0,1].plot(
        baseline["loss"],
        label="Training"
    )

    axes[0,1].plot(
        baseline["val_loss"],
        label="Validation"
    )


    axes[0,1].set_title(
        "Baseline CNN Loss"
    )

    axes[0,1].set_xlabel(
        "Epoch"
    )

    axes[0,1].set_ylabel(
        "Loss"
    )

    axes[0,1].legend()



    # Optimized Accuracy

    axes[1,0].plot(
        optimized["accuracy"],
        label="Training"
    )

    axes[1,0].plot(
        optimized["val_accuracy"],
        label="Validation"
    )


    axes[1,0].set_title(
        "Optimized CNN Accuracy"
    )

    axes[1,0].set_xlabel(
        "Epoch"
    )

    axes[1,0].set_ylabel(
        "Accuracy"
    )

    axes[1,0].legend()



    # Optimized Loss

    axes[1,1].plot(
        optimized["loss"],
        label="Training"
    )

    axes[1,1].plot(
        optimized["val_loss"],
        label="Validation"
    )


    axes[1,1].set_title(
        "Optimized CNN Loss"
    )

    axes[1,1].set_xlabel(
        "Epoch"
    )

    axes[1,1].set_ylabel(
        "Loss"
    )

    axes[1,1].legend()


    plt.tight_layout()


    plt.savefig(
        OUTPUT_DIRECTORY /
        "training_comparison.png",
        dpi=300
    )

    plt.close()



if __name__ == "__main__":

    plot_training_comparison()