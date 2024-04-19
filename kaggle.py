import os
import cv2
import numpy as np
from keras.models import load_model

# Define constants and parameters
IMG_SIZE = 200  # Define the size of the images
LABELS = {0: 'NORMAL', 1: 'PNEUMONIA'}  # Mapping from index to label

# Load the for_training model
model = load_model('kaggle.keras')


# Load and preprocess the image
def load_and_preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize image
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = img / 255.0  # Normalize pixel values
    return img


# Path to the folder containing validation images
val_folder = 'for_testing'

# Iterate over subfolders (NORMAL and PNEUMONIA)
for label in LABELS.values():
    label_folder = os.path.join(val_folder, label)
    if os.path.isdir(label_folder):
        # Iterate over images in the subfolder
        for img_name in os.listdir(label_folder):
            if img_name.startswith('.'):
                continue  # Skip hidden files
            img_path = os.path.join(label_folder, img_name)

            # Load and preprocess the image
            image = load_and_preprocess_image(img_path)

            # Reshape the input data to match the expected input shape of the model
            image = np.expand_dims(image, axis=0)  # Add batch dimension

            # Make prediction using the loaded model
            prediction = model.predict(image)
            predicted_class = np.argmax(prediction)

            # Get the predicted label
            predicted_label = LABELS[predicted_class]

            # Print the results
            print(f'Image: {img_name}, Predicted class: {predicted_label}')
