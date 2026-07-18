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

from training_visualization import plot_training_history

from model_evaluation import evaluate_model

from optimized_cnn import (
    build_optimized_cnn,
    compile_optimized_cnn,
)

def main():

    # Set the data directory path and directory setup
    data_directory = 'homework_three/Fish'
    split_directory = "homework_three/data/splits"

    project_setup.create_homework_three_structure()

    # This section is for obtaining the datasets details

    file_counts = count_file_types(data_directory)

    print("\nDataset File Types")
    print("------------------")

    for file_type, count in file_counts.items():
        print(f"{file_type}: {count}")

    class_counts = count_classes(data_directory)

    print("\nFish Classes")
    print("------------")

    for class_name, count in class_counts.items():
        print(f"{class_name}: {count}")

    total_images = sum(class_counts.values())

    print(f"\nTotal Classes: {len(class_counts)}")
    print(f"Total Images: {total_images}")

    image_properties = inspect_image_properties(data_directory)

    print("\nImage Dimensions")
    print("----------------")

    for dimensions, count in image_properties["dimensions"].items():
        print(f"{dimensions[0]} x {dimensions[1]}: {count}")

    print("\nColor Modes")
    print("-----------")

    for color_mode, count in image_properties["color_modes"].items():
        print(f"{color_mode}: {count}")

    print("\nCorrupted Images")
    print("----------------")

    corrupted_images = image_properties["corrupted_images"]

    if corrupted_images:
        for corrupted_image in corrupted_images:
            print(
                f"{corrupted_image['file']}: "
                f"{corrupted_image['error']}"
            )
    else:
        print("No corrupted images found.")

    split_data = create_stratified_split(
        data_directory=data_directory,
        output_directory=split_directory,
    )

    display_split_summary(split_data)

    split_csv = (
        "homework_three/data/splits/"
        "dataset_splits.csv"
    )

    (
        train_dataset,
        validation_dataset,
        test_dataset,
        class_mapping,
    ) = create_datasets(
        split_csv
    )

    number_of_classes = len(
        class_mapping
    )

    # baseline_model = build_baseline_cnn(
    #     number_of_classes
    # )

    # baseline_model = compile_baseline_cnn(
    #     baseline_model
    # )

    # baseline_model.summary()

    # baseline_output_directory = (
    #     "homework_three/outputs/baseline"
    # )
    

    # # Train baseline only once
    # history = train_baseline_model(
    #     model=baseline_model,
    #     train_dataset=train_dataset,
    #     validation_dataset=validation_dataset,
    #     output_directory=baseline_output_directory,
    #     epochs=30,
    # )

    # # Plot the history from that same training run
    # plot_training_history(
    #     history=history,
    #     output_directory=baseline_output_directory,
    # )

    # # Load the best saved model
    # saved_model = load_saved_model(
    #     "homework_three/outputs/baseline/best_baseline_model.keras"
    # )

    # # Evaluate the best saved model on the untouched test set
    # baseline_results = evaluate_model(
    #     model=saved_model,
    #     test_dataset=test_dataset,
    #     class_mapping=class_mapping,
    # )

    optimized_model = build_optimized_cnn(
        number_of_classes
    )

    optimized_model = compile_optimized_cnn(
        optimized_model
    )

    optimized_model.summary()


    optimized_output_directory = (
        "homework_three/outputs/optimized"
    )


    optimized_history = train_model(
        model=optimized_model,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        output_directory=optimized_output_directory,
        epochs=30,
    )


    plot_training_history(
        history=optimized_history,
        output_directory=optimized_output_directory,
    )


    optimized_model_path = (
        "homework_three/outputs/optimized/"
        "best_optimized_model.keras"
    )


    saved_optimized_model = load_saved_model(
        optimized_model_path
    )


    optimized_results = evaluate_model(
        model=saved_optimized_model,
        test_dataset=test_dataset,
        class_mapping=class_mapping,
    )

    # Uncomment this section to visualize the augmentations applied to the training dataset.

    # visualize_augmentations(
    #     train_dataset,
    #     class_mapping,
    # )

    print("\nClass Mapping")
    print("-------------")

    for class_name, class_id in class_mapping.items():
        print(
            f"{class_name}: {class_id}"
        )

    # for images, labels in train_dataset.take(1):
    #     print("\nTraining Batch")
    #     print("--------------")
    #     print(
    #         f"Image batch shape: {images.shape}"
    #     )
    #     print(
    #         f"Label batch shape: {labels.shape}"
    #     )
    #     print(
    #         f"Minimum pixel value: {tf.reduce_min(images).numpy():.4f}"
    #     )
    #     print(
    #         f"Maximum pixel value: {tf.reduce_max(images).numpy():.4f}"
    #     )


if __name__ == "__main__":
    main()