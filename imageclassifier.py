import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2 as cs
import numpy as np
from keras import models, layers, datasets
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()

# Normalize images
training_images, testing_images = training_images / 255.0, testing_images / 255.0

# Class names
className = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Display sample images
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(className[training_labels[i][0]])

plt.show()

# Reduce the dataset size for faster training (optional)
training_images, training_labels = training_images[:20000], training_labels[:20000]
testing_images, testing_labels = testing_images[:4000], testing_labels[:4000]

# Build the model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(training_images, training_labels, epochs=50, validation_data=(testing_images, testing_labels))

# Evaluate the model
loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

# Save the model
model.save('D:/Program/Python/Practise/AI/Practise/Image Classification/Image Classification.keras')
model.save('D:/Program/Python/Practise/AI/Practise/Image Classification/Image Classification.h5')