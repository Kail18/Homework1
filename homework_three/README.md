# Homework 3: Deep Learning for Fish Classification

## Project Overview

This project focuses on building, optimizing, and evaluating Convolutional Neural Networks (CNNs) for multiclass fish image classification.

The completed workflow includes:

- Project directory setup
- Dataset inspection and validation
- Stratified train/validation/test splitting
- Image preprocessing and normalization
- Class label mapping
- Training data augmentation
- Baseline CNN development and evaluation
- Optimized CNN development and evaluation
- Early stopping and model checkpointing
- Quantitative evaluation using accuracy, precision, recall, and F1-score
- Per-class classification reports
- Confusion matrix analysis
- Training and validation curve comparison

---

## Dataset

The dataset contains six fish classes.

| Class     | Number of Images |
| --------- | ---------------: |
| Oscar     |              145 |
| Cray      |               80 |
| Guppy     |              189 |
| Bete      |              194 |
| Gold      |              207 |
| Discuss   |              201 |
| **Total** |         **1016** |

The dataset is moderately imbalanced. The smallest class, `Cray`, contains 80 images, while the largest class, `Gold`, contains 207 images.

Because of this imbalance, a stratified train/validation/test split was used to preserve approximately the same class proportions across all three subsets.

---

## Dataset File Types

The dataset contains the following image formats:

| File Type |    Count |
| --------- | -------: |
| `.jpg`    |      962 |
| `.png`    |       54 |
| **Total** | **1016** |

---

```text
800 x 600 pixels
```

| Color Mode | Count |
| ---------- | ----: |
| RGB        |  1014 |
| RGBA       |     2 |

The preprocessing pipeline converts all images to three-channel RGB format before they are passed to the CNN.

No corrupted images were detected during the dataset audit.

---

## Train, Validation, and Test Split

The dataset was divided using a stratified split:

- Training: 70%
- Validation: 15%
- Testing: 15%
  A fixed random seed of 42 is used to make the split reproducible.
  The final split contains:
  | Split | Number of Images |
  |---|---:|
  | Training | 711 |
  | Validation | 152 |
  | Testing | 153 |
  | **Total** | **1016** |

### Per-Class Split Distribution

| Class   | Training | Validation | Testing | Total |
| ------- | -------: | ---------: | ------: | ----: |
| Bete    |      136 |         29 |      29 |   194 |
| Cray    |       56 |         12 |      12 |    80 |
| Discuss |      141 |         30 |      30 |   201 |
| Gold    |      145 |         31 |      31 |   207 |
| Guppy   |      132 |         28 |      29 |   189 |
| Oscar   |      101 |         22 |      22 |   145 |

The stratified split ensures that each class remains represented in the training, validation, and test datasets.
The split assignments are saved to:

```text
dataset_splits.csv
```

This CSV acts as the single source of truth for all future model training and evaluation.
Both the baseline CNN and optimized CNN will use the exact same train, validation, and test assignments.

---

## Class Mapping

The fish class labels are converted into numeric class IDs before being passed to the CNN.
| Class | Class ID |
|---|---:|
| Bete | 0 |
| Cray | 1 |
| Discuss | 2 |
| Gold | 3 |
| Guppy | 4 |
| Oscar | 5 |
This mapping is kept consistent across training, validation, testing, and final evaluation.

---

## Image Preprocessing

All images are processed before being passed into the CNN.
The preprocessing pipeline performs the following steps:

1. Load the image from disk
2. Convert the image to three-channel RGB
3. Resize the image from 800 x 600 to 128 x 128
4. Convert pixel values to floating-point numbers
5. Normalize pixel values from the original 0–255 range to 0–1

The final image tensor shape is:

```text
128 x 128 x 3
```

Will consider 224 x 224 if the initial model struggles to distinguish visually similar species. In lecture we discussed trying to use lower resolutions during the early HPO to reduce computational cost.

For a batch size of 32, the CNN receives image batches with the shape:

```text
(32, 128, 128, 3)
```

The preprocessing pipeline was validated with the following output:

```text
Minimum pixel value: 0.0000
Maximum pixel value: 1.0000
```

This confirms that the normalization process successfully converts image intensity values to the required [0, 1] range.

---

## Data Augmentation

Data augmentation is applied only to the training images.
The current augmentation pipeline includes:

- Random horizontal flipping
- Small random rotations
- Random contrast adjustment

These transformations introduce controlled variation into the training dataset.
The goal is to improve model generalization by reducing the likelihood that the CNN memorizes specific orientations, lighting conditions, or image characteristics.
Validation and test images are not randomly augmented.
This ensures that model performance is measured using consistent, unmodified evaluation data.

### Data Augmentation Analysis

Random horizontal flipping, small rotations, and random contrast adjustments were applied only during training. These augmentations introduced variation in fish orientation and image appearance without modifying the validation or test datasets.

The purpose of these transformations was to reduce the model's dependence on specific image orientations and lighting conditions. Training accuracy fluctuated more when augmentation and Dropout were active in the optimized model because the model received slightly different transformed images during each epoch. However, these transformations also made memorization more difficult and contributed to the smaller gap observed between training and validation accuracy.

The augmentation pipeline was intentionally kept moderate. Large rotations or aggressive transformations were avoided because they could produce unrealistic fish orientations or distort class-specific visual features.

## Baseline Model

### Baseline CNN Hyperparameters

| Hyperparameter          | Value                              |
| ----------------------- | ---------------------------------- |
| Input Image Size        | `128 x 128 x 3`                    |
| Batch Size              | `32`                               |
| Maximum Epochs          | `30`                               |
| Optimizer               | `Adam`                             |
| Learning Rate           | `0.001`                            |
| Loss Function           | `Sparse Categorical Cross-Entropy` |
| Number of Classes       | `6`                                |
| Conv Layer 1 Filters    | `32`                               |
| Conv Layer 2 Filters    | `64`                               |
| Conv Layer 3 Filters    | `128`                              |
| Convolution Kernel Size | `3 x 3`                            |
| Padding                 | `same`                             |
| Activation Function     | `ReLU`                             |
| Max Pool Size           | `2 x 2`                            |
| Dense Layer Units       | `256`                              |
| Output Activation       | `Softmax`                          |
| Early Stopping Monitor  | `val_loss`                         |
| Early Stopping Patience | `5` epochs                         |
| Restore Best Weights    | `True`                             |
| Checkpoint Monitor      | `val_loss`                         |
| Save Best Model Only    | `True`                             |
| Random Seed             | `42`                               |
| Training Split          | `70%`                              |
| Validation Split        | `15%`                              |
| Test Split              | `15%`                              |
| Image Normalization     | `[0, 1]`                           |
| Horizontal Flip         | `Enabled`                          |
| Random Rotation         | `0.05`                             |
| Random Contrast         | `0.20`                             |

### Baseline Training Analysis

The baseline CNN showed strong learning performance on the training dataset. Training accuracy steadily increased to approximately 98%, while training loss decreased to approximately 0.06.

Validation performance initially improved, reaching approximately 85–86% accuracy. However, validation accuracy eventually plateaued and declined while validation loss began to increase. This divergence between training and validation performance indicates overfitting.

The model checkpoint monitored validation loss and preserved the model from the epoch with the best validation performance rather than using the final training epoch. These results suggest that the baseline model has excessive capacity relative to the dataset size and may benefit from additional regularization, dropout, or reduced model complexity.

### Baseline Model Results

#### Baseline Metrics

| Metric        |  Score |
| ------------- | -----: |
| Test Loss     | 0.6539 |
| Test Accuracy | 83.66% |
| Precision     | 84.88% |
| Recall        | 83.66% |
| F1-Score      | 82.96% |

#### Baseline Class Performance

| Class   | Precision | Recall | F1-Score |
| ------- | --------: | -----: | -------: |
| Bete    |      0.78 |   0.86 |     0.82 |
| Cray    |      0.80 |   0.33 |     0.47 |
| Discuss |      0.90 |   0.93 |     0.92 |
| Gold    |      1.00 |   0.84 |     0.91 |
| Guppy   |      0.72 |   0.97 |     0.82 |
| Oscar   |      0.85 |   0.77 |     0.81 |

### Hyperparameter optimization / Improve CNN architecture

- Replace Flatten() with GlobalAveragePooling2D() (Makes the model focus on what features exist rather than where they appear)
- Add Dropout (to help force the model to learn more generalized patterns vs specific patterns from the training images)
- Reduce parameter count (I dont need a big network since the dataset is small)
- Tune learning rate and batch size (attempt to improve stability and create better generalizations)

| Metric               |       Result |
| -------------------- | -----------: |
| Training Accuracy    |         ~98% |
| Validation Accuracy  |         ~85% |
| Test Accuracy        |       83.66% |
| Trainable Parameters | 8.48 million |
| Training Images      |          711 |

With these metrics I do not need to improve the models memorization. The main thing I need to focus on is generalization.

## Optimized CNN Architecture

The optimized CNN was developed to address the overfitting observed in the baseline model. Since the dataset contains only 711 training images, the goal was not to increase model capacity but to improve generalization.

The following modifications were applied:

| Modification                              | Purpose                                                                         |
| ----------------------------------------- | ------------------------------------------------------------------------------- |
| GlobalAveragePooling2D instead of Flatten | Reduced parameter count and encouraged learning spatial feature representations |
| Dropout layers                            | Reduced neuron dependency and improved generalization                           |
| Data augmentation                         | Increased training diversity and reduced memorization                           |
| Reduced fully connected layer size        | Prevented excessive parameters relative to dataset size                         |
| Early stopping                            | Prevented unnecessary training after validation performance stopped improving   |

### Optimized CNN Hyperparameters

| Hyperparameter         |                            Value |
| ---------------------- | -------------------------------: |
| Input Image Size       |                    128 x 128 x 3 |
| Batch Size             |                               64 |
| Maximum Epochs         |                               30 |
| Optimizer              |                             Adam |
| Learning Rate          |                            0.001 |
| Loss Function          | Sparse Categorical Cross-Entropy |
| Conv Layer 1 Filters   |                               32 |
| Conv Layer 2 Filters   |                               64 |
| Conv Layer 3 Filters   |                              128 |
| Global Average Pooling |                          Enabled |
| Dropout                |                          Enabled |
| Dense Layer Units      |                               64 |
| Output Classes         |                                6 |

## Optimized Model Training Analysis

The optimized CNN was designed to reduce the overfitting observed in the baseline model by replacing the large Flatten-based classification head with GlobalAveragePooling2D, reducing the dense layer size, and adding Dropout regularization.

These changes substantially reduced the model from approximately 8.48 million trainable parameters to 101,894 trainable parameters.

The optimized model showed a much smaller gap between training and validation accuracy than the baseline model, indicating that overfitting was reduced. However, both training and validation accuracy remained substantially lower than those of the baseline model. This suggests that the optimized architecture may have been over-regularized or lacked sufficient capacity to learn all of the discriminative features required for the six fish classes.

The lowest validation loss was achieved at epoch 20. Training continued until epoch 25, when early stopping was triggered, and the model weights from epoch 20 were restored.

Although the optimized architecture successfully reduced model complexity and overfitting, it did not improve performance on the held-out test dataset.

### Optimized Training Results

| Metric                   |   Score |
| ------------------------ | ------: |
| Selected Batch Size      |      64 |
| Best Validation Loss     |  0.9690 |
| Best Validation Accuracy |  65.79% |
| Test Loss                |  0.8969 |
| Test Accuracy            |  68.63% |
| Precision                |  66.83% |
| Recall                   |  68.63% |
| F1-Score                 |  66.11% |
| Trainable Parameters     | 101,894 |

#### Optimized Class Performance

| Class   | Precision | Recall | F1-Score |
| ------- | --------: | -----: | -------: |
| Bete    |      0.68 |   0.52 |     0.59 |
| Cray    |      0.00 |   0.00 |     0.00 |
| Discuss |      1.00 |   0.90 |     0.95 |
| Gold    |      0.88 |   0.90 |     0.89 |
| Guppy   |      0.47 |   0.90 |     0.62 |
| Oscar   |      0.53 |   0.41 |     0.46 |

## Baseline vs Optimized Model Comparison

| Model         | Test Accuracy | Precision | Recall | F1-Score |
| ------------- | ------------: | --------: | -----: | -------: |
| Baseline CNN  |        83.66% |    84.88% | 83.66% |   82.96% |
| Optimized CNN |        68.63% |    66.83% | 68.63% |   66.11% |

### Comparison Discussion

The optimized CNN substantially reduced model complexity and showed less separation between training and validation performance, indicating that the regularization techniques were effective at reducing overfitting. However, this improvement came at the cost of classification performance.

The baseline CNN achieved the strongest held-out test results, with 83.66% accuracy and an F1-score of 82.96%, compared with 66.01% accuracy and a 66.28% F1-score for the optimized model.

The results suggest that replacing the large Flatten-based classification head with GlobalAveragePooling2D, reducing the dense layer size, and applying two Dropout layers may have reduced model capacity too aggressively. The optimized model appears to have shifted from an overfitting problem toward an underfitting problem.

The baseline therefore remains the better-performing model for this dataset, although its training curves indicate that additional regularization could still improve its generalization. A future model could use a less aggressive combination of regularization techniques, such as lower Dropout rates or a larger dense classification layer.

### Batch Size Hyperparameter Optimization

Two batch-size configurations were evaluated using the optimized CNN architecture. All other major training settings, including the learning rate, dataset split, random seed, and maximum number of epochs, were kept constant. The configurations were compared using validation performance rather than the held-out test dataset.

| Batch Size | Best Validation Loss | Best Validation Accuracy |
| ---------- | -------------------: | -----------------------: |
| 32         |               1.0404 |                   63.16% |
| 64         |               0.9690 |                   65.79% |

Batch size 64 produced both a lower validation loss and higher validation accuracy. Because early stopping and model checkpointing were based on validation loss, batch size 64 was selected for the final optimized model.

The larger batch size also reduced the number of training steps per epoch from approximately 23 steps with batch size 32 to 12 steps with batch size 64. In this experiment, batch size 64 provided more stable validation performance and improved final test accuracy from 66.01% in the earlier optimized run to 68.63%.

Although accuracy and recall improved, weighted precision and F1-score remained similar. This demonstrates that improving one evaluation metric does not necessarily improve performance uniformly across all classes.

## Optimized CNN Confusion Matrix

The confusion matrix below shows the classification performance of the optimized CNN across all six fish species.

![Optimized CNN Confusion Matrix](outputs/optimized/confusion_matrix.png)

The Cray class was the most difficult class for the final optimized model, with both precision and recall equal to 0.00. This means that none of the 12 Cray images in the held-out test set were correctly classified. Cray also contains the fewest examples in the dataset, with only 56 training images, which may contribute to the model's difficulty learning robust features for this class. Visual similarities between Cray and other classes may also contribute to these classification errors.

The model performs strongest on classes such as Gold and Discuss, which contain more training examples and have more visually distinctive characteristics.

## Training Curve Comparison

The following visualization compares baseline and optimized CNN training behavior.

![CNN Training Comparison](outputs/model_comparison/training_comparison.png)

# Conclusion

The baseline CNN achieved the strongest classification performance but showed evidence of overfitting due to the gap between training and validation performance. The optimized CNN substantially reduced model complexity through Global Average Pooling and Dropout, decreasing the trainable parameter count from approximately 8.48 million to 101,894. These changes reduced the degree of overfitting, but also reduced overall classification performance, suggesting that the optimized architecture may have been over-regularized or lacked sufficient capacity.

Batch-size hyperparameter testing compared configurations of 32 and 64. Batch size 64 achieved the lower validation loss and higher validation accuracy and was therefore selected for the final optimized model. The final optimized model achieved 68.63% test accuracy compared with 83.66% for the baseline model.

The largest limitation of this project is the relatively small and imbalanced dataset. The Cray class, which contains the fewest training examples, remained particularly difficult for the optimized model. Future improvements could include less aggressive Dropout, a larger classification head, class weighting, additional training data, transfer learning with a pretrained CNN, or more targeted domain-specific augmentation.
