# Homework 3: Deep Learning for Fish Classification

## Project Overview

This project focuses on building and evaluating a Convolutional Neural Network (CNN) for multiclass fish image classification.

The current work completed includes:

- Project directory setup
- Dataset inspection and validation
- Stratified train/validation/test splitting
- Image preprocessing
- Class label mapping
- Training data augmentation
- Augmentation visualization

The next major step will be building and training the baseline CNN.

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
