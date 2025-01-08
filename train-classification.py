# For data manipulation
import numpy as np
# import pandas as pd

# For data visualization
import matplotlib.pyplot as plt
# import seaborn as sns

# Ingore the warnings
import warnings
warnings.filterwarnings('ignore')

# DL Libraries
import tensorflow
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, AveragePooling2D
# from tensorflow.keras.utils import image_dataset_from_directory
from keras.preprocessing import image_dataset_from_directory
# Other libraries
import os
import random


train_ds = keras.preprocessing.image_dataset_from_directory(
    directory = r'./brain-tumor-mri-dataset/Training',
    labels='inferred',
    label_mode='int',
    batch_size=64,
    image_size=(256, 256),
)

val_ds = keras.preprocessing.image_dataset_from_directory(
    directory = r'./brain-tumor-mri-dataset/Testing',
    labels='inferred',
    label_mode='int',
    batch_size=64,
    image_size=(256, 256),
)


# Creating a function to visualize the images

def visualize_images(path, num_images=5):

    # Get a list of image filenames
    image_filenames = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    if not image_filenames:
        raise ValueError("No images found in the specified path")

    # Select random images
    selected_images = random.sample(image_filenames, min(num_images, len(image_filenames)))

    # Create a figure and axes
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3), facecolor='white')

    # Display each image
    for i, image_filename in enumerate(selected_images):
        # Load image
        image_path = os.path.join(path, image_filename)
        image = plt.imread(image_path)
        # Display image
        axes[i].imshow(image)
        axes[i].axis('off')
        axes[i].set_title(image_filename)  # Set image filename as title

        # Adjust layout and display
    plt.tight_layout()
    plt.show()

# Extrating the class labels
classes = train_ds.class_names

    # Iterating through each class to plot its images
for label in classes:
        # Specify the path containing the images to visualize
        path_to_visualize = f"./brain-tumor-mri-dataset/Training/{label}"

        # Visualize 3 random images
        print(label.upper())
        visualize_images(path_to_visualize, num_images=4)

model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.1))

model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.1))

model.add(Dense(64, activation='relu'))

# model.add(Dropout(0.2))

model.add(Dense(4, activation='softmax'))

model.summary()

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Train the model

history = model.fit(train_ds, epochs=10, validation_data=val_ds, verbose=1)

model.save('model-4.h5')  # 保存为 HDF5 格式

# from keras.models import load_model
#
# history = load_model('model-4.h5')

# Plotting the graph of Accuracy and Validation Accuracy
plt.title('Training Accuracy vs Validation Accuracy')

plt.plot(history.history['accuracy'], color='red',label='Train')
plt.plot(history.history['val_accuracy'], color='blue',label='Validation')

plt.legend()
plt.show()


# Plotting the graph of Accuracy and Validation loss
plt.title('Training Loss vs Validation Loss')

plt.plot(history.history['loss'], color='red',label='Train')
plt.plot(history.history['val_loss'], color='blue',label='Validation')

plt.legend()
plt.show()
