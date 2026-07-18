import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


def evaluate_model(
    model: tf.keras.Model,
    test_dataset: tf.data.Dataset,
    class_mapping: dict,
) -> dict:
    """
    Evaluate a trained model on the untouched test dataset.

    Calculates:
    - Test loss
    - Test accuracy
    - Weighted precision
    - Weighted recall
    - Weighted F1-score
    - Classification report

    Args:
        model:
            Trained Keras model.

        test_dataset:
            TensorFlow test dataset.

        class_mapping:
            Dictionary mapping class names to integer labels.

    Returns:
        Dictionary containing evaluation metrics.
    """

    # Standard Keras test evaluation
    test_loss, test_accuracy = model.evaluate(
        test_dataset,
        verbose=1,
    )

    true_labels = []
    predicted_labels = []

    # Generate predictions for every test batch
    for images, labels in test_dataset:

        predictions = model.predict(
            images,
            verbose=0,
        )

        predicted_classes = np.argmax(
            predictions,
            axis=1,
        )

        true_labels.extend(
            labels.numpy()
        )

        predicted_labels.extend(
            predicted_classes
        )

    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    # Calculate classification metrics
    accuracy = accuracy_score(
        true_labels,
        predicted_labels,
    )

    precision = precision_score(
        true_labels,
        predicted_labels,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        true_labels,
        predicted_labels,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        true_labels,
        predicted_labels,
        average="weighted",
        zero_division=0,
    )

    # Reverse mapping:
    # 0 -> Bete
    # 1 -> Cray
    # etc.
    inverse_class_mapping = {
        label: class_name
        for class_name, label
        in class_mapping.items()
    }

    class_names = [
        inverse_class_mapping[index]
        for index in range(len(inverse_class_mapping))
    ]

    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names,
        zero_division=0,
    )

    print("\nBaseline Test Results")
    print("---------------------")
    print(f"Test Loss:      {test_loss:.4f}")
    print(f"Test Accuracy:  {accuracy:.4f}")
    print(f"Precision:      {precision:.4f}")
    print(f"Recall:         {recall:.4f}")
    print(f"F1-Score:       {f1:.4f}")

    print("\nClassification Report")
    print("---------------------")
    print(report)

    return {
        "test_loss": test_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_labels": true_labels,
        "predicted_labels": predicted_labels,
        "classification_report": report,
    }