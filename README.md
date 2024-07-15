Here's a more professional and organized version of your README:

---

# Plant Coverage Rate Estimation Using Classification

## Introduction

This project was part of my internship at Shanghai Roots & Shoots: Million Tree Project, a non-profit organization that plants trees in Inner Mongolia, China. The goal is to use drone images of tree plantations to estimate plant coverage rates to help better monitor tree growth. I used an SVM model to classify the pixels based on RGB values into two categories: "tree" and "soil", and used the percentage of "tree" pixels as the plant coverage rate for each image.

### Original Image
<img src="https://user-images.githubusercontent.com/80374850/200893490-f7614751-1e76-47ed-9508-0130eafed159.JPG" width="400">

### Image After Classification
![clf_DJI_0206](https://user-images.githubusercontent.com/80374850/200893564-9533c752-ef20-45c8-9322-67b100ef73af.JPG)

## Preprocessing

Because the images were taken under different lighting conditions, the color tone differs from image to image. This makes it difficult to generalize model results to unseen images. To solve this problem, I centered the RGB values of each image so that all the images would have the same mean value for each of the RGB channels.

### Unprocessed Image
<img src="https://user-images.githubusercontent.com/80374850/200897450-d8afcc6a-a182-4b1a-9f97-60e427fe880c.JPG" width="400">

### Processed Image
<img src="https://user-images.githubusercontent.com/80374850/200897502-321ce376-1977-4e45-a683-39dd822db182.JPG" width="400">

## Model Training

I took screenshots of soil and tree pixels to use as training data and trained an SVM model to classify the pixels.

## Results

The final SVM classification model achieved an accuracy of 0.927. The estimated plant coverage rates for the test images are shown below.

![Plant Coverage Rates](https://user-images.githubusercontent.com/80374850/200925360-ebd54961-d877-4bc4-b04f-eba68fb67e18.png)

## Explanation of Python Files

### `image_preprocessing.py`

Preprocesses images by:
1. Centering the RGB values
2. Reducing the size of the image

**Input:**
- Folder containing unprocessed images
- Constant k between 0 and 1, which is the proportion used to reduce the image size

**Output:**
- Processed images saved to the folder "Processed_Images"

### `model_training.py`

Trains a classification model.

**Input:**
- Screenshots of soil and trees from images used as training data

**Output:**
- Classification model saved as `classification_model.sav`

### `tree_coverage.py`

Performs classification and calculates plant coverage rate.

**Input:**
- Folder containing preprocessed images

**Output:**
- "Results" folder containing:
  - Plant coverage rates (in CSV and Excel formats)
  - Visualization of images after classification

---
