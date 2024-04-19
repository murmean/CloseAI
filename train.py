import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define constants and parameters
DATA_DIR = 'for_training'  # Path to your dataset directory
IMG_SIZE = 200  # Define the size of the images
NUM_CLASSES = 2  # Number of classes: pneumonia and normal

# Load and preprocess the data
def load_data(data_dir):
    data = []
    labels = []
    label_to_index = {'NORMAL': 0, 'PNEUMONIA': 1}  # Define mapping from label to index
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):  # Skip if it's not a directory
            continue
        for img_name in os.listdir(label_dir):
            if img_name.startswith('.'):  # Skip hidden files
                continue
            img_path = os.path.join(label_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize image
            data.append(img)
            labels.append(label_to_index[label])  # Use numerical encoding
    data = np.array(data) / 255.0  # Normalize pixel values
    labels = np.array(labels)
    return data, labels

# Load data
X, y = load_data(DATA_DIR)

# Convert labels to one-hot encoding
y = to_categorical(y, num_classes=NUM_CLASSES)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dense(NUM_CLASSES, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model on all the data
history = model.fit(X.reshape(-1, IMG_SIZE, IMG_SIZE, 1), y,
                    epochs=10, batch_size=64)

# Save the for_training model
model.save("kaggle.keras")

# Visualize training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
