import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, models

# Load CIFAR-10 dataset
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
training_images, testing_images = training_images / 255.0, testing_images / 255.0

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)  # Turn off grid
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])  # Fix the xlabel assignment
plt.show()

# Limit the dataset size for demonstration
training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:40000]
testing_labels = testing_labels[:40000]

# Load the pre-for_training model
model = models.load_model('image_classifier.h5')

# Attempt to load and process the image
try:
    img = cv.imread('frog.jpg')
    if img is None:
        raise FileNotFoundError("Failed to load image. Check file path and file existence.")
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.imshow(img, cmap=plt.cm.binary)
    plt.show()
    # Perform prediction
    prediction = model.predict(np.expand_dims(img / 255.0, axis=0))
    index = np.argmax(prediction)
    print(f'Prediction is {class_names[index]}')

except Exception as e:
    print(f"An error occurred: {e}")
