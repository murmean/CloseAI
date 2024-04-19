import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from keras.models import load_model

# Define constants and parameters
IMG_SIZE = 200  # Define the size of the images
LABELS = {0: 'NORMAL', 1: 'PNEUMONIA'}  # Mapping from index to label

# Load the trained model
model = load_model('kaggle.keras')

def load_and_preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize image
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = img / 255.0  # Normalize pixel values
    return img

def predict_image(image_path):
    image = load_and_preprocess_image(image_path)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    predicted_label = LABELS[predicted_class]
    return predicted_label

def browse_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        predicted_label = predict_image(file_path)
        result_label.config(text=f'The predicted class is: {predicted_label}')

# Create the main application window
root = tk.Tk()
root.title("Pneumonia Prediction App")

# Create a frame for the UI elements
frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

# Create a button to browse for an image
browse_button = tk.Button(frame, text="Browse Image", command=browse_image)
browse_button.pack(side=tk.LEFT)

# Create a label to display the prediction result
result_label = tk.Label(frame, text="")
result_label.pack(side=tk.LEFT)

# Run the application
root.mainloop()
