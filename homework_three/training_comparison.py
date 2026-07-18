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


    figure = plt.figure(
        figsize=(14,14)
    )

    grid = figure.add_gridspec(
        3,
        2
    )

    ax1 = figure.add_subplot(grid[0,0])
    ax2 = figure.add_subplot(grid[0,1])
    ax3 = figure.add_subplot(grid[1,0])
    ax4 = figure.add_subplot(grid[1,1])
    ax5 = figure.add_subplot(grid[2,:])


    # Baseline Accuracy

    ax1.plot(
        baseline["accuracy"],
        label="Training"
    )

    ax1.plot(
        baseline["val_accuracy"],
        label="Validation"
    )

    ax1.set_title(
        "Baseline CNN Accuracy"
    )

    ax1.set_xlabel(
        "Epoch"
    )

    ax1.set_ylabel(
        "Accuracy"
    )

    ax1.legend()



    # Baseline Loss

    ax2.plot(
        baseline["loss"],
        label="Training"
    )

    ax2.plot(
        baseline["val_loss"],
        label="Validation"
    )


    ax2.set_title(
        "Baseline CNN Loss"
    )

    ax2.set_xlabel(
        "Epoch"
    )

    ax2.set_ylabel(
        "Loss"
    )

    ax2.legend()



    # Optimized Accuracy

    ax3.plot(
        optimized["accuracy"],
        label="Training"
    )

    ax3.plot(
        optimized["val_accuracy"],
        label="Validation"
    )


    ax3.set_title(
        "Optimized CNN Accuracy"
    )

    ax3.set_xlabel(
        "Epoch"
    )

    ax3.set_ylabel(
        "Accuracy"
    )

    ax3.legend()



    # Optimized Loss

    ax4.plot(
        optimized["loss"],
        label="Training"
    )

    ax4.plot(
        optimized["val_loss"],
        label="Validation"
    )


    ax4.set_title(
        "Optimized CNN Loss"
    )

    ax4.set_xlabel(
        "Epoch"
    )

    ax4.set_ylabel(
        "Loss"
    )

    ax4.legend()

    true_labels = np.load(
        "homework_three/outputs/optimized/true_labels.npy"
    )

    predicted_labels = np.load(
        "homework_three/outputs/optimized/predicted_labels.npy"
    )


    cm = confusion_matrix(
        true_labels,
        predicted_labels
    )


    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        ax=ax5,
        cmap="Blues"
    )


    ax5.set_title(
        "Optimized CNN Confusion Matrix"
    )

    ax5.set_xlabel(
        "Predicted Class"
    )

    ax5.set_ylabel(
        "True Class"
    )


    plt.tight_layout()


    plt.savefig(
        OUTPUT_DIRECTORY /
        "training_comparison.png",
        dpi=300
    )

    plt.close()



if __name__ == "__main__":

    plot_training_comparison()