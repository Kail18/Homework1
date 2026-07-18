# Prompts will be done with ChatGPT

## Prompt 1 15:26 7/16/2026

**Prompt:** Lets come up with a stepwise plan to start completing this project.

**Response:**
Framework: PyTorch
Image size: Begin with 128 × 128
Dataset split: 70% training, 15% validation, 15% testing
Random seed: 42
Baseline optimizer: Adam
Baseline learning rate: 0.001
Baseline batch size: 32
Tuning strategy: Small grid search
Model-selection metric: Lowest validation loss
Primary evaluation: Accuracy, precision, recall, F1-score, and confusion matrix

## Prompt 2 8:34 7/18/2026

**Prompt:**
This is the output Dataset File Types

---

.jpg: 962
.png: 54

## Fish Classes

Oscar: 145
Cray: 80
Guppy: 189
Bete: 194
Gold: 207
Discuss: 201

Total Classes: 6
Total Images: 1016

**Response:**
It recommended another function to inspect the image_properties()

## Prompt 3 8:36 7/18/2026

**Prompt:**
ok lets add that function

**Response:**
I added the output function to data_details

## Prompt 4 8:40 7/18/2026

**Prompt:**
Dataset File Types

---

.jpg: 962
.png: 54

## Fish Classes

Oscar: 145
Cray: 80
Guppy: 189
Bete: 194
Gold: 207
Discuss: 201

Total Classes: 6
Total Images: 1016

## Image Dimensions

800 x 600: 1016

## Color Modes

RGB: 1014
RGBA: 2

## Corrupted Images

No corrupted images found.

This is my updated outputs

**Response:**
It recommended using a Strtified train/validation/test split secondary to the uneven distribution of classes. This is to prevent the random split from accidently placing too few Cray images in either validation or testing. With the stratisfied I would have roughly the same class proportions in each subset.

This was a good recommendation as I was already considering this fact from what we discussed in the lecture Computer Vision Optimization

## Prompt 5 8:51 7/18/2026

**Prompt:**

I agree with this assessment. Lets start by creating the stratified/train/validation/test split for the my six classes.

**Response:**

It provided a data_split.py file.
One thing to add it had an output to data/splits as a CSV to help keep the project reproducible. Also the CSV was to prevent data leakage and use for a simplier data loading later.

## Prompt 6 9:13 7/18/2026

**Prompt:**

## Dataset File Types

.jpg: 962
.png: 54

## Fish Classes

Oscar: 145
Cray: 80
Guppy: 189
Bete: 194
Gold: 207
Discuss: 201

Total Classes: 6
Total Images: 1016

## Image Dimensions

800 x 600: 1016

## Color Modes

RGB: 1014
RGBA: 2

## Corrupted Images

No corrupted images found.

## Dataset Split Summary

split test train validation total
class_name  
Bete 29 136 29 194
Cray 12 56 12 80
Discuss 30 141 30 201
Gold 31 145 31 207
Guppy 29 132 28 189
Oscar 22 101 22 145

## Total Images Per Split

split
train 711
test 153
validation 152
Name: count, dtype: int64

Here is the updated output

**Response:**

It agreed with my assessment that stratified split was working.

## Prompt 7 9:15 7/18/2026

**Prompt:**

Lets continue to the next step. Lets do the image preprocessing and augmentation.

**Response:**
Preprocessing plan
Image
↓
Load JPG or PNG
↓
Force 3-channel RGB
↓
Resize from 800×600 → 128×128
↓
Convert pixel values from 0–255 → 0–1

I added the recommended image_preprocessing code. It recommended using tensorflow

## Prompt 3 8:36 7/18/2026

**Prompt:**

## Dataset File Types

.jpg: 962
.png: 54

## Fish Classes

Oscar: 145
Cray: 80
Guppy: 189
Bete: 194
Gold: 207
Discuss: 201

Total Classes: 6
Total Images: 1016

## Image Dimensions

800 x 600: 1016

## Color Modes

RGB: 1014
RGBA: 2

## Corrupted Images

No corrupted images found.

## Dataset Split Summary

split test train validation total
class_name  
Bete 29 136 29 194
Cray 12 56 12 80
Discuss 30 141 30 201
Gold 31 145 31 207
Guppy 29 132 28 189
Oscar 22 101 22 145

## Total Images Per Split

split
train 711
test 153
validation 152
Name: count, dtype: int64

## Class Mapping

Bete: 0
Cray: 1
Discuss: 2
Gold: 3
Guppy: 4
Oscar: 5

## Training Batch

Image batch shape: (32, 128, 128, 3)
Label batch shape: (32,)
Minimum pixel value: 0.0000
Maximum pixel value: 1.0000

This is the updated output

**Response:**

It agreed with the expected output. One thing it recommended adding was adding a way to double check the preprocessing. Currently the print functions only tells us the preprocessing was successful. I agree with adding some visualization for the few training images.

## Prompt 3 10:17 7/18/2026

**Prompt:**

Lets move onto the baseline CNN model.

**Response:**

It provided a basic setup for the model that I include in model.py file.

### Recommended Convolution Layer

```text
tf.keras.layers.Conv2D(
    filters=32,
    kernel_size=(3, 3),
    activation="relu",
    padding="same",
)
```

This is done so the model starts learning the basic features. The deeper layers will start learning the more complex features.

my image will progress:

```text
128 × 128
    ↓
64 × 64
    ↓
32 × 32
    ↓
16 × 16
```

while the feature maps will increase:

```text
3
↓
32
↓
64
↓
128
```

### Output layer

```text
tf.keras.layers.Dense(
    number_of_classes,
    activation="softmax",
)
```

Softmax is used to convert the outputs into probabilities with the sum = 1.0.

## Prompt 3 10:25 7/18/2026

**Prompt:**

## Dataset File Types

.jpg: 962
.png: 54

## Fish Classes

Oscar: 145
Cray: 80
Guppy: 189
Bete: 194
Gold: 207
Discuss: 201

Total Classes: 6
Total Images: 1016

## Image Dimensions

800 x 600: 1016

## Color Modes

RGB: 1014
RGBA: 2

## Corrupted Images

No corrupted images found.

## Dataset Split Summary

split test train validation total
class_name  
Bete 29 136 29 194
Cray 12 56 12 80
Discuss 30 141 30 201
Gold 31 145 31 207
Guppy 29 132 28 189
Oscar 22 101 22 145

## Total Images Per Split

split
train 711
test 153
validation 152
Name: count, dtype: int64
Model: "baseline_fish_cnn"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Layer (type) ┃ Output Shape ┃ Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ data_augmentation │ (None, 128, 128, 3) │ 0 │
│ (Sequential) │ │ │
├────────────────────────────┼─────────────────────┼─────────────┤
│ conv2d (Conv2D) │ (None, 128, 128, │ 896 │
│ │ 32) │ │
├────────────────────────────┼─────────────────────┼─────────────┤
│ max_pooling2d │ (None, 64, 64, 32) │ 0 │
│ (MaxPooling2D) │ │ │
├────────────────────────────┼─────────────────────┼─────────────┤
│ conv2d_1 (Conv2D) │ (None, 64, 64, 64) │ 18,496 │
├────────────────────────────┼─────────────────────┼─────────────┤
│ max_pooling2d_1 │ (None, 32, 32, 64) │ 0 │
│ (MaxPooling2D) │ │ │
├────────────────────────────┼─────────────────────┼─────────────┤
│ conv2d_2 (Conv2D) │ (None, 32, 32, 128) │ 73,856 │
├────────────────────────────┼─────────────────────┼─────────────┤
│ max_pooling2d_2 │ (None, 16, 16, 128) │ 0 │
│ (MaxPooling2D) │ │ │
├────────────────────────────┼─────────────────────┼─────────────┤
│ flatten (Flatten) │ (None, 32768) │ 0 │
├────────────────────────────┼─────────────────────┼─────────────┤
│ dense (Dense) │ (None, 256) │ 8,388,864 │
├────────────────────────────┼─────────────────────┼─────────────┤
│ dense_1 (Dense) │ (None, 6) │ 1,542 │
└────────────────────────────┴─────────────────────┴─────────────┘
Total params: 8,483,654 (32.36 MB)
Trainable params: 8,483,654 (32.36 MB)
Non-trainable params: 0 (0.00 B)
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [0.0..1.0351955].
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [0.0..1.085335].
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [0.0..1.1573204].
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [0.0..1.0030719].
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [0.0..1.1412904].
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [0.0..1.0039997].
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [0.0..1.0997386].
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [0.0..1.1011589].
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [0.0..1.1257879].
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [0.0..1.169305].
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [0.0..1.0510343].
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [0.0..1.1320156].

## Class Mapping

Bete: 0
Cray: 1
Discuss: 2
Gold: 3
Guppy: 4
Oscar: 5

## Training Batch

Image batch shape: (32, 128, 128, 3)
Label batch shape: (32,)
Minimum pixel value: 0.0000
Maximum pixel value: 1.0000

Here are the updated outputs

**Response:**

Discussed the:
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Got range [0.0..1.169305].

I added some code to the visualization to fix the clipping.

It mentioned a large amount of parameters for only 711 images so I will keep this in consideration to watch for overfitting.

## Prompt 3 10:33 7/18/2026

**Prompt:**

Lets do a baseline train with an early stopping model checkpoint.

**Response:**

I implented code to the train.py

key notes there are 30 epochs with an early stopping:

```text
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1,
)
```

## Prompt 3 11:09 7/18/2026

**Prompt:**

Next generate the training/validation loss and accuracy curves.

**Response:**

provided a script for training_visualization.py script.

## Prompt 3 11:24 7/18/2026

**Prompt:**

Evaluate the baseline on the untouched test set

**Response:**

provided a script for model_evaluation.py

## Prompt 3 11:54 7/18/2026

**Prompt:**

lets start the optimization of the CNN architecture
Replace Flatten() with GlobalAveragePooling2D()
Add Dropout
Reduce parameter count
Tune learning rate and batch size

**Response:**

Recommended creating a new optimized_cnn so that we can compare to the baseline model

## Prompt 3 12:52 7/18/2026

**Prompt:**

I provided my current README and had it update the things I was missing.

**Response:**

I used the updated README prompts where needed and rewrote some of the sections to better reflect what I observed.

Template

## Prompt 3 8:36 7/18/2026

**Prompt:**

**Response:**
