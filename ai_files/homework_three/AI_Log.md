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

Template

## Prompt 3 8:36 7/18/2026

**Prompt:**

**Response:**
