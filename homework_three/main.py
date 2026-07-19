import json
import shutil
from pathlib import Path

import project_setup
import tensorflow as tf

from data_details import (
    count_file_types,
    count_classes,
    inspect_image_properties,
)

from data_split import (
    create_stratified_split,
    display_split_summary,
)

from image_preprocessing import (
    create_datasets,
    visualize_augmentations,
)

from model import (
    build_baseline_cnn,
    compile_baseline_cnn,
    load_saved_model,
)

from train import train_model

from training_visualization import (
    plot_training_history,
    plot_model_comparison,
)

from model_evaluation import (
    evaluate_model,
)

from optimized_cnn import (
    build_optimized_cnn,
    compile_optimized_cnn,
)


def main():

    # =============================================
    # Project and Dataset Paths
    # =============================================

    data_directory = "homework_three/Fish"

    split_directory = (
        "homework_three/data/splits"
    )

    project_setup.create_homework_three_structure()

    # =============================================
    # Dataset Details
    # =============================================

    file_counts = count_file_types(
        data_directory
    )

    print("\nDataset File Types")
    print("------------------")

    for file_type, count in file_counts.items():
        print(
            f"{file_type}: {count}"
        )

    class_counts = count_classes(
        data_directory
    )

    print("\nFish Classes")
    print("------------")

    for class_name, count in class_counts.items():
        print(
            f"{class_name}: {count}"
        )

    total_images = sum(
        class_counts.values()
    )

    print(
        f"\nTotal Classes: "
        f"{len(class_counts)}"
    )

    print(
        f"Total Images: "
        f"{total_images}"
    )

    image_properties = inspect_image_properties(
        data_directory
    )

    print("\nImage Dimensions")
    print("----------------")

    for dimensions, count in image_properties[
        "dimensions"
    ].items():

        print(
            f"{dimensions[0]} x "
            f"{dimensions[1]}: "
            f"{count}"
        )

    print("\nColor Modes")
    print("-----------")

    for color_mode, count in image_properties[
        "color_modes"
    ].items():

        print(
            f"{color_mode}: "
            f"{count}"
        )

    print("\nCorrupted Images")
    print("----------------")

    corrupted_images = image_properties[
        "corrupted_images"
    ]

    if corrupted_images:

        for corrupted_image in corrupted_images:

            print(
                f"{corrupted_image['file']}: "
                f"{corrupted_image['error']}"
            )

    else:

        print(
            "No corrupted images found."
        )

    # =============================================
    # Create Stratified Dataset Split
    # =============================================

    split_data = create_stratified_split(
        data_directory=data_directory,
        output_directory=split_directory,
    )

    display_split_summary(
        split_data
    )

    split_csv = (
        "homework_three/data/splits/"
        "dataset_splits.csv"
    )

    # =============================================
    # Create Baseline Datasets
    # Baseline uses batch size 32
    # =============================================

    (
        train_dataset,
        validation_dataset,
        test_dataset,
        class_mapping,
    ) = create_datasets(
        split_csv,
        batch_size=32,
    )

    number_of_classes = len(
        class_mapping
    )

    # =============================================
    # Baseline CNN
    # =============================================

    tf.keras.utils.set_random_seed(
        42
    )

    baseline_model = build_baseline_cnn(
        number_of_classes
    )

    baseline_model = compile_baseline_cnn(
        baseline_model
    )

    baseline_model.summary()

    baseline_output_directory = (
        "homework_three/outputs/baseline"
    )

    # Train baseline model
    baseline_history = train_model(
        model=baseline_model,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        output_directory=baseline_output_directory,
        epochs=30,
    )

    # Plot baseline training history
    plot_training_history(
        history=baseline_history,
        output_directory=baseline_output_directory,
        model_name="Baseline CNN",
    )

    # Load best baseline checkpoint
    saved_baseline_model = load_saved_model(
        "homework_three/outputs/baseline/"
        "best_model.keras"
    )

    # Evaluate baseline on held-out test set
    baseline_results = evaluate_model(
        model=saved_baseline_model,
        test_dataset=test_dataset,
        class_mapping=class_mapping,
        output_directory=baseline_output_directory,
        model_name="Baseline CNN",
    )

    # =============================================
    # Grid Search Hyperparameter Optimization
    # =============================================

    learning_rates = [
        0.01,
        0.001,
        0.0001,
    ]

    batch_sizes = [
        32,
        64,
    ]

    dropout_rates = [
        0.3,
        0.5,
    ]

    hpo_results = []

    best_validation_loss = float(
        "inf"
    )

    best_configuration = None
    best_model_path = None
    best_history = None

    configuration_number = 0

    total_configurations = (
        len(learning_rates)
        * len(batch_sizes)
        * len(dropout_rates)
    )

    # =============================================
    # Run Grid Search
    # =============================================

    for learning_rate in learning_rates:

        for batch_size in batch_sizes:

            for dropout_rate in dropout_rates:

                configuration_number += 1

                print(
                    "\n============================================="
                )

                print(
                    f"Hyperparameter Configuration "
                    f"{configuration_number} "
                    f"of {total_configurations}"
                )

                print(
                    "============================================="
                )

                print(
                    f"Learning Rate: "
                    f"{learning_rate}"
                )

                print(
                    f"Batch Size: "
                    f"{batch_size}"
                )

                print(
                    f"Dropout Rate: "
                    f"{dropout_rate}"
                )

                # =============================================
                # Create Datasets for Current Batch Size
                # =============================================

                (
                    experiment_train_dataset,
                    experiment_validation_dataset,
                    _,
                    _,
                ) = create_datasets(
                    split_csv,
                    batch_size=batch_size,
                )

                # Clear previous TensorFlow model state
                tf.keras.backend.clear_session()

                # Reset random seed so each configuration
                # starts from a reproducible state
                tf.keras.utils.set_random_seed(
                    42
                )

                # =============================================
                # Build Current HPO Model
                # =============================================

                experiment_model = (
                    build_optimized_cnn(
                        number_of_classes,
                        dropout_rate=dropout_rate,
                    )
                )

                experiment_model = (
                    compile_optimized_cnn(
                        experiment_model,
                        learning_rate=learning_rate,
                    )
                )

                configuration_name = (
                    f"lr_{learning_rate}_"
                    f"batch_{batch_size}_"
                    f"dropout_{dropout_rate}"
                )

                experiment_output_directory = (
                    "homework_three/outputs/hpo/"
                    f"{configuration_name}"
                )

                # =============================================
                # Train Current Configuration
                # =============================================

                experiment_history = train_model(
                    model=experiment_model,
                    train_dataset=(
                        experiment_train_dataset
                    ),
                    validation_dataset=(
                        experiment_validation_dataset
                    ),
                    output_directory=(
                        experiment_output_directory
                    ),
                    epochs=30,
                )

                validation_losses = (
                    experiment_history.history[
                        "val_loss"
                    ]
                )

                validation_accuracies = (
                    experiment_history.history[
                        "val_accuracy"
                    ]
                )

                # Find the epoch with the lowest
                # validation loss
                best_epoch_index = min(
                    range(
                        len(
                            validation_losses
                        )
                    ),
                    key=(
                        validation_losses.__getitem__
                    ),
                )

                configuration_validation_loss = (
                    validation_losses[
                        best_epoch_index
                    ]
                )

                configuration_validation_accuracy = (
                    validation_accuracies[
                        best_epoch_index
                    ]
                )

                configuration_best_epoch = (
                    best_epoch_index + 1
                )

                print(
                    "\nConfiguration Results"
                )

                print(
                    "---------------------"
                )

                print(
                    f"Best Epoch: "
                    f"{configuration_best_epoch}"
                )

                print(
                    f"Best Validation Loss: "
                    f"{configuration_validation_loss:.4f}"
                )

                print(
                    f"Validation Accuracy at "
                    f"Best Epoch: "
                    f"{configuration_validation_accuracy:.4f}"
                )

                # =============================================
                # Save Configuration Results
                # =============================================

                configuration_results = {
                    "learning_rate":
                        learning_rate,

                    "batch_size":
                        batch_size,

                    "dropout_rate":
                        dropout_rate,

                    "best_epoch":
                        configuration_best_epoch,

                    "best_val_loss":
                        configuration_validation_loss,

                    "val_accuracy_at_best_epoch":
                        configuration_validation_accuracy,
                }

                hpo_results.append(
                    configuration_results
                )

                # =============================================
                # Track Global Best Configuration
                # =============================================

                if (
                    configuration_validation_loss
                    < best_validation_loss
                ):

                    best_validation_loss = (
                        configuration_validation_loss
                    )

                    best_configuration = {
                        "learning_rate":
                            learning_rate,

                        "batch_size":
                            batch_size,

                        "dropout_rate":
                            dropout_rate,

                        "best_epoch":
                            configuration_best_epoch,

                        "best_val_loss":
                            configuration_validation_loss,

                        "val_accuracy_at_best_epoch":
                            configuration_validation_accuracy,
                    }

                    best_model_path = (
                        Path(
                            experiment_output_directory
                        )
                        / "best_model.keras"
                    )

                    best_history = (
                        experiment_history
                    )

    # =============================================
    # Save Complete HPO Results
    # =============================================

    hpo_output_directory = Path(
        "homework_three/outputs/hpo"
    )

    hpo_output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    hpo_results_path = (
        hpo_output_directory
        / "hpo_results.json"
    )

    with open(
        hpo_results_path,
        "w",
    ) as file:

        json.dump(
            hpo_results,
            file,
            indent=4,
        )

    # =============================================
    # Display Grid Search Results
    # =============================================

    print(
        "\n============================================="
    )

    print(
        "Grid Search HPO Results"
    )

    print(
        "============================================="
    )

    for index, result in enumerate(
        hpo_results,
        start=1,
    ):

        print(
            f"\nConfiguration {index}"
        )

        print(
            f"Learning Rate: "
            f"{result['learning_rate']}"
        )

        print(
            f"Batch Size: "
            f"{result['batch_size']}"
        )

        print(
            f"Dropout Rate: "
            f"{result['dropout_rate']}"
        )

        print(
            f"Best Validation Loss: "
            f"{result['best_val_loss']:.4f}"
        )

        print(
            f"Validation Accuracy at "
            f"Best Epoch: "
            f"{result['val_accuracy_at_best_epoch']:.4f}"
        )

        print(
            f"Best Epoch: "
            f"{result['best_epoch']}"
        )

    # =============================================
    # Display Best Hyperparameter Configuration
    # =============================================

    print(
        "\n============================================="
    )

    print(
        "Best Hyperparameter Configuration"
    )

    print(
        "============================================="
    )

    print(
        f"Learning Rate: "
        f"{best_configuration['learning_rate']}"
    )

    print(
        f"Batch Size: "
        f"{best_configuration['batch_size']}"
    )

    print(
        f"Dropout Rate: "
        f"{best_configuration['dropout_rate']}"
    )

    print(
        f"Best Epoch: "
        f"{best_configuration['best_epoch']}"
    )

    print(
        f"Best Validation Loss: "
        f"{best_configuration['best_val_loss']:.4f}"
    )

    print(
        f"Validation Accuracy at Best Epoch: "
        f"{best_configuration['val_accuracy_at_best_epoch']:.4f}"
    )

    # =============================================
    # Save Best Configuration Information
    # =============================================

    best_configuration_path = (
        hpo_output_directory
        / "best_configuration.json"
    )

    with open(
        best_configuration_path,
        "w",
    ) as file:

        json.dump(
            best_configuration,
            file,
            indent=4,
        )

    # =============================================
    # Save Best HPO Model as Final Optimized Model
    # =============================================

    optimized_output_directory = Path(
        "homework_three/outputs/optimized"
    )

    optimized_output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_optimized_model_path = (
        optimized_output_directory
        / "best_model.keras"
    )

    shutil.copy2(
        best_model_path,
        final_optimized_model_path,
    )

    print(
        "\nBest HPO model saved to: "
        f"{final_optimized_model_path}"
    )

    # =============================================
    # Create Datasets Using Best Batch Size
    # =============================================

    (
        optimized_train_dataset,
        optimized_validation_dataset,
        optimized_test_dataset,
        optimized_class_mapping,
    ) = create_datasets(
        split_csv,
        batch_size=(
            best_configuration[
                "batch_size"
            ]
        ),
    )

    # =============================================
    # Load Final Optimized Model
    # =============================================

    saved_optimized_model = load_saved_model(
        str(
            final_optimized_model_path
        )
    )

    saved_optimized_model.summary()

    # =============================================
    # Plot Best Optimized Training History
    # =============================================

    plot_training_history(
        history=best_history,
        output_directory=str(
            optimized_output_directory
        ),
        model_name="Optimized CNN",
    )

    # =============================================
    # Evaluate Final Optimized Model
    # Test set is used only after HPO is complete
    # =============================================

    optimized_results = evaluate_model(
        model=saved_optimized_model,
        test_dataset=optimized_test_dataset,
        class_mapping=optimized_class_mapping,
        output_directory=str(
            optimized_output_directory
        ),
        model_name="Optimized CNN",
    )

    # =============================================
    # Combined Model Comparison Visualization
    # =============================================

    plot_model_comparison(
        baseline_history=baseline_history,
        optimized_history=best_history,
        optimized_confusion_matrix_path=(
            "homework_three/outputs/optimized/"
            "confusion_matrix.png"
        ),
        output_directory=(
            "homework_three/outputs/"
            "model_comparison"
        ),
    )

    # =============================================
    # Optional Augmentation Visualization
    # =============================================

    # visualize_augmentations(
    #     train_dataset,
    #     class_mapping,
    # )

    # =============================================
    # Class Mapping
    # =============================================

    print(
        "\nClass Mapping"
    )

    print(
        "-------------"
    )

    for (
        class_name,
        class_id,
    ) in class_mapping.items():

        print(
            f"{class_name}: "
            f"{class_id}"
        )

    # =============================================
    # Optional Batch Validation
    # =============================================

    # for images, labels in train_dataset.take(1):
    #
    #     print("\nTraining Batch")
    #     print("--------------")
    #
    #     print(
    #         f"Image batch shape: "
    #         f"{images.shape}"
    #     )
    #
    #     print(
    #         f"Label batch shape: "
    #         f"{labels.shape}"
    #     )
    #
    #     print(
    #         f"Minimum pixel value: "
    #         f"{tf.reduce_min(images).numpy():.4f}"
    #     )
    #
    #     print(
    #         f"Maximum pixel value: "
    #         f"{tf.reduce_max(images).numpy():.4f}"
    #     )


if __name__ == "__main__":
    main()