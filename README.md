# Vehicle Classifier with Convolutional Neural Network (CNN)

## Overview

This project utilizes a Convolutional Neural Network (CNN) to classify images of vehicles as either cars or trucks. The deep learning model is implemented using TensorFlow, and the dataset consists of images of cars and trucks. The project involves preprocessing the data, building and training the CNN model, evaluating its performance, and making predictions on new images.

## Installation and Setup

1. **Install Dependencies:**

    ```bash
    !pip install tensorflow opencv-python matplotlib
    ```

2. **Import Libraries and Remove Dodgy Images:**

    ```python
    import tensorflow as tf
    import os
    import cv2
    import imghdr
    import numpy as np
    from matplotlib import pyplot as plt
    ```

## Data Preprocessing

1. **Remove Dodgy Images:**

   Remove images with unsupported extensions from the dataset.

2. **Load Data:**

   Use TensorFlow's `image_dataset_from_directory` to load the dataset. Visualize a sample batch of images.

3. **Scale Data:**
   
   Scale pixel values of images to the range [0, 1].

4. **Split Data:**
   
   Split the dataset into training, validation, and test sets.

## Model Architecture

 **Build Deep Learning Model:**
 
   Construct a CNN model using the Sequential API. The model includes convolutional layers, max-pooling layers, and dense layers with ReLU activation. The output layer uses a sigmoid activation for binary classification.

## Training

1. **Train:**
   
   Train the model using the training set and validate on the validation set. Monitor training progress using TensorBoard logs.

2. **Plot Performance:**
   
   Visualize the training and validation loss, as well as accuracy, over epochs.

## Evaluation

 **Evaluate:**
 
  Evaluate the model on the test set using precision, recall, and binary accuracy metrics.

## Testing

 **Test:**
 
  Test the model on a new image. Display the image, resize it, make a prediction, and print the predicted class.

## Save the Model

 **Save the Model:**
 
  Save the trained model for future use.

## Usage

To use this project, follow the steps outlined in the provided Python script. Ensure that the required dependencies are installed, and the dataset is structured according to the specified directory. The trained model can be saved and loaded for making predictions on new images.

Feel free to customize the model architecture, hyperparameters, or extend the dataset to improve performance.

For any issues or questions, please refer to the documentation or contact the project contributors.

**Note:** Please replace 'data' and 'test/car.jpeg' with the actual paths to your dataset and test image.

This README provides a concise overview of the project, and users can refer to the Python script for detailed implementation.
