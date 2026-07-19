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

| Split      | Number of Images |
| ---------- | ---------------: |
| Training   |              711 |
| Validation |              152 |
| Testing    |              153 |
| **Total**  |         **1016** |

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

### Optimization Strategy

The baseline CNN achieved strong classification performance but showed evidence of overfitting. Training accuracy approached 98%, while validation performance plateaued at a lower level. Therefore, the optimization process focused on reducing overfitting and improving generalization rather than increasing the model's ability to memorize the training data.

The following architecture changes were introduced:

- Replaced `Flatten()` with `GlobalAveragePooling2D()` to substantially reduce the number of trainable parameters.
- Reduced the fully connected hidden layer from 256 units to 64 units.
- Added Dropout regularization to reduce neuron dependency and discourage memorization.
- Retained data augmentation to expose the model to variations in orientation and contrast.
- Used early stopping and model checkpointing based on validation loss.

A systematic Grid Search was then performed over three hyperparameters:

- Learning rate: `0.01`, `0.001`, and `0.0001`
- Batch size: `32` and `64`
- Dropout rate: `0.3` and `0.5`

This produced a total of 12 hyperparameter configurations. Each configuration was evaluated using validation loss, and the held-out test dataset was not used for hyperparameter selection.

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
| Batch Size             |                               32 |
| Maximum Epochs         |                               30 |
| Optimizer              |                             Adam |
| Learning Rate          |                            0.001 |
| Loss Function          | Sparse Categorical Cross-Entropy |
| Conv Layer 1 Filters   |                               32 |
| Conv Layer 2 Filters   |                               64 |
| Conv Layer 3 Filters   |                              128 |
| Global Average Pooling |                          Enabled |
| Dropout Rate           |                              0.5 |
| Dense Layer Units      |                               64 |
| Output Classes         |                                6 |

## Optimized Model Training Analysis

The optimized CNN was designed to reduce the overfitting observed in the baseline model by replacing the large Flatten-based classification head with `GlobalAveragePooling2D`, reducing the dense layer size, and adding Dropout regularization.

These architecture changes reduced the model from approximately 8.48 million trainable parameters to 101,894 trainable parameters.

A Grid Search was performed across 12 combinations of learning rate, batch size, and Dropout rate. The best-performing configuration was selected using the lowest validation loss.

The selected configuration used:

- Learning rate: `0.001`
- Batch size: `32`
- Dropout rate: `0.5`

The best configuration reached its lowest validation loss of 0.9704 at epoch 17, with a validation accuracy of 65.79%.

The hyperparameter experiments showed that the learning rate had a substantial effect on convergence. A learning rate of 0.01 generally produced higher validation losses, suggesting that the optimization steps were too aggressive for this architecture. A learning rate of 0.0001 improved slowly and generally required the full 30 epochs, suggesting that convergence was too slow within the available training budget. The intermediate learning rate of 0.001 produced the best overall validation performance.

For the winning learning rate and batch-size combination, a Dropout rate of 0.5 achieved a lower validation loss than 0.3. The final selected batch size was 32.

Although the optimized architecture reduced model complexity and showed less evidence of severe overfitting, its test performance remained below that of the baseline CNN. This suggests that the combination of Global Average Pooling, a smaller dense layer, and Dropout may have reduced the model's capacity too aggressively, resulting in underfitting.

### Optimized Model Results

| Metric                            |   Score |
| --------------------------------- | ------: |
| Selected Learning Rate            |   0.001 |
| Selected Batch Size               |      32 |
| Selected Dropout Rate             |     0.5 |
| Best Epoch                        |      17 |
| Best Validation Loss              |  0.9704 |
| Validation Accuracy at Best Epoch |  65.79% |
| Test Loss                         |  0.9346 |
| Test Accuracy                     |  67.97% |
| Precision                         |  67.89% |
| Recall                            |  67.97% |
| F1-Score                          |  65.04% |
| Trainable Parameters              | 101,894 |

#### Optimized Class Performance

| Class   | Precision | Recall | F1-Score |
| ------- | --------: | -----: | -------: |
| Bete    |      0.70 |   0.66 |     0.68 |
| Cray    |      0.00 |   0.00 |     0.00 |
| Discuss |      0.93 |   0.87 |     0.90 |
| Gold    |      0.81 |   0.81 |     0.81 |
| Guppy   |      0.47 |   0.93 |     0.62 |
| Oscar   |      0.78 |   0.32 |     0.45 |

## Baseline vs Optimized Model Comparison

| Model         | Test Accuracy | Precision | Recall | F1-Score |
| ------------- | ------------: | --------: | -----: | -------: |
| Baseline CNN  |        83.66% |    84.88% | 83.66% |   82.96% |
| Optimized CNN |        67.97% |    67.89% | 67.97% |   65.04% |

### Comparison Discussion

The optimized CNN substantially reduced model complexity, decreasing the trainable parameter count from approximately 8.48 million to 101,894. The smaller model and added Dropout regularization reduced the severe training and validation separation observed in the baseline model. However, this reduction in model capacity came at the cost of classification performance.

The baseline CNN achieved the strongest held-out test results, with 83.66% accuracy and an F1-score of 82.96%. The optimized CNN achieved 67.97% accuracy and an F1-score of 65.04%.

These results suggest that replacing the Flatten-based classification head with GlobalAveragePooling2D, reducing the dense layer from 256 to 64 units, and applying a Dropout rate of 0.5 may have reduced model capacity too aggressively. The optimization process reduced overfitting, but the resulting model appears to have shifted toward underfitting.

Therefore, the baseline CNN remains the strongest-performing model for this dataset. A future architecture could seek a middle ground between the two models, such as retaining Global Average Pooling while increasing the dense layer size or using less aggressive regularization.

### Grid Search Hyperparameter Optimization

A systematic Grid Search was used to evaluate three hyperparameters:

- Learning rate: `0.01`, `0.001`, `0.0001`
- Batch size: `32`, `64`
- Dropout rate: `0.3`, `0.5`

This resulted in 12 total configurations.

| Configuration | Learning Rate | Batch Size | Dropout | Best Validation Loss | Validation Accuracy at Best Epoch | Best Epoch |
| ------------- | ------------: | ---------: | ------: | -------------------: | --------------------------------: | ---------: |
| 1             |          0.01 |         32 |     0.3 |               1.7525 |                            19.74% |          5 |
| 2             |          0.01 |         32 |     0.5 |               1.4283 |                            41.45% |         10 |
| 3             |          0.01 |         64 |     0.3 |               1.2367 |                            52.63% |         20 |
| 4             |          0.01 |         64 |     0.5 |               1.3974 |                            36.18% |          7 |
| 5             |         0.001 |         32 |     0.3 |               1.0544 |                            61.84% |         12 |
| 6             |         0.001 |         32 |     0.5 |           **0.9704** |                        **65.79%** |     **17** |
| 7             |         0.001 |         64 |     0.3 |               1.0570 |                            63.82% |         21 |
| 8             |         0.001 |         64 |     0.5 |               1.0717 |                            58.55% |         26 |
| 9             |        0.0001 |         32 |     0.3 |               1.3262 |                            44.74% |         30 |
| 10            |        0.0001 |         32 |     0.5 |               1.3913 |                            44.74% |         30 |
| 11            |        0.0001 |         64 |     0.3 |               1.4193 |                            41.45% |         30 |
| 12            |        0.0001 |         64 |     0.5 |               1.4771 |                            42.76% |         30 |

The best configuration was selected based on the lowest validation loss:

```text
Learning Rate: 0.001
Batch Size: 32
Dropout Rate: 0.5
Best Epoch: 17
Best Validation Loss: 0.9704
Validation Accuracy at Best Epoch: 65.79%
```

## Optimized CNN Confusion Matrix

The confusion matrix below shows the classification performance of the optimized CNN across all six fish species.

![Optimized CNN Confusion Matrix](outputs/optimized/confusion_matrix.png)

The Cray class was the most difficult class for the final optimized model, with both precision and recall equal to 0.00. This means that none of the 12 Cray images in the held-out test set were correctly classified. Cray also contains the fewest examples in the dataset, with only 56 training images, which may contribute to the model's difficulty learning robust features for this class. Visual similarities between Cray and other classes may also contribute to these classification errors.

The model performs strongest on classes such as Gold and Discuss, which contain more training examples and have more visually distinctive characteristics.

## Training Curve Comparison

The following visualization compares the training and validation accuracy and loss of the baseline and optimized CNNs. The optimized model's confusion matrix is included to show its final multiclass classification performance.

![CNN Training Comparison](outputs/model_comparison/training_comparison.png)

# Conclusion

This project demonstrated a complete deep-learning image classification workflow, including dataset inspection, stratified data splitting, preprocessing, augmentation, custom CNN development, systematic hyperparameter tuning, and quantitative model evaluation.

The baseline CNN achieved the strongest classification performance, reaching 83.66% test accuracy and an F1-score of 82.96%. However, its training and validation curves showed evidence of overfitting, with training performance continuing to improve while validation performance began to plateau or decline.

The optimized CNN reduced the trainable parameter count from approximately 8.48 million to 101,894 by replacing Flatten with Global Average Pooling, reducing the dense classification layer, and introducing Dropout regularization.

A Grid Search evaluated 12 combinations of learning rate, batch size, and Dropout rate. The best configuration used a learning rate of 0.001, batch size of 32, and Dropout rate of 0.5. This configuration achieved the lowest validation loss of 0.9704 at epoch 17.

Despite reducing model complexity and limiting overfitting, the final optimized model achieved only 67.97% test accuracy and a 65.04% F1-score. This indicates that the optimization strategy likely reduced model capacity too aggressively and shifted the model toward underfitting. Therefore, the baseline CNN remained the better-performing classifier.

The largest limitation of this project is the relatively small and imbalanced dataset. The Cray class contains only 56 training images and remained particularly difficult for the optimized model, which failed to correctly classify any Cray examples in the held-out test set.

Future improvements could explore a less aggressive balance between model capacity and regularization, such as increasing the dense layer size, testing lower Dropout rates, applying class weighting, collecting additional training images, using transfer learning with a pretrained CNN, or introducing more targeted domain-specific image augmentation.
