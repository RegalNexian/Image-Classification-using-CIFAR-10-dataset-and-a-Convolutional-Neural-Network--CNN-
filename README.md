# Image Classification using CIFAR-10 dataset and a Convolutional Neural Network (CNN)

This repository contains an image classification project using the CIFAR-10 dataset and a Convolutional Neural Network (CNN) implemented with Keras. The model is trained to classify images into one of ten categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Training the Model](#training-the-model)
- [Using the Pre-trained Model](#using-the-pre-trained-model)
- [Acknowledgements](#acknowledgements)

## Introduction

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. This project demonstrates how to build, train, and evaluate a CNN on this dataset using Keras, and then use the trained model to make predictions on new images.

## Requirements

- Python 3.6 or higher
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/image-classification.git
    cd image-classification
    ```

2. Create and activate a virtual environment (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Training the Model

To train the model from scratch, run the following script:

```bash
python train_model.py
