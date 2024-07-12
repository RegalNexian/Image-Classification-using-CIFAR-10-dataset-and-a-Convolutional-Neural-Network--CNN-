import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2 as cv
import numpy as np
from keras import models, layers, datasets
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()

# Normalize images
training_images, testing_images = training_images / 255.0, testing_images / 255.0

# Class names
className = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']


# Reduce the dataset size for faster training (optional)
training_images, training_labels = training_images[:20000], training_labels[:20000]
testing_images, testing_labels = testing_images[:4000], testing_labels[:4000]

# Load the model
model_path = 'D:/Program/Python/Practise/AI/Practise/Image Classification/Image Classification.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

model = models.load_model(model_path)

# Read and preprocess the image
img_path = 'D:/Program/Python/Practise/AI/Practise/Image Classification/download (3) (1).jpeg'
if not os.path.exists(img_path):
    raise FileNotFoundError(f"Image file not found at: {img_path}")

img1 = cv.imread(img_path)
if img1 is None:
    raise FileNotFoundError(f"Image file not found at: {img_path}")

# Convert the image from BGR to RGB
img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)

# Resize the image to the input size of the model (32x32)
img1 = cv.resize(img1, (32, 32))

# Display the image
plt.imshow(img1, cmap=plt.cm.binary)
plt.show()

# Predict the class of the image
img1 = np.expand_dims(img1, axis=0)  # Add batch dimension
img1 = img1 / 255.0  # Normalize the image

pred = model.predict(img1)
index = np.argmax(pred)

print(f"Prediction is {className[index]}")
