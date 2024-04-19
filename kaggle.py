import cv2
import numpy as np
from keras.models import load_model

# Define constants and parameters
IMG_SIZE = 200  # Define the size of the images
LABELS = {0: 'NORMAL', 1: 'PNEUMONIA'}  # Mapping from index to label

# Load the trained model
model = load_model('kaggle.keras')

# Load and preprocess the image
def load_and_preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize image
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = img / 255.0  # Normalize pixel values
    return img

# Path to the image you want to make predictions on
image_path = 'val/PNEUMONIA/person1946_bacteria_4875.jpeg'  # Replace with the path to your image

# Load and preprocess the image
image = load_and_preprocess_image(image_path)

# Reshape the input data to match the expected input shape of the model
image = np.expand_dims(image, axis=0)  # Add batch dimension

# Make prediction
prediction = model.predict(image)
predicted_class = np.argmax(prediction)

# Get the label corresponding to the predicted class
predicted_label = LABELS[predicted_class]

print(f'The predicted class is: {predicted_label}')