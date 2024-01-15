

# Non-essential
import pandas as pd



input_path = "/Users/MayraBerrones/Documents/DDSM-CIBIS/Full_enhance"


import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Set the seed for reproducibility
# Set the seed for reproducibility
tf.compat.v1.set_random_seed(42)

# Define the U-Net model
def unet(input_shape):
    inputs = keras.Input(shape=input_shape)
    
    # Downsample path
    conv1 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = keras.layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = keras.layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = keras.layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
    pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = keras.layers.Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = keras.layers.Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    
    # Upsample path
    up6 = keras.layers.Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(conv5)
    up6 = keras.layers.concatenate([up6, conv4])
    conv6 = keras.layers.Conv2D(512, 3, activation='relu', padding='same')(up6)
    conv6 = keras.layers.Conv2D(512, 3, activation='relu', padding='same')(conv6)
    
    up7 = keras.layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv6)
    up7 = keras.layers.concatenate([up7, conv3])
    conv7 = keras.layers.Conv2D(256, 3, activation='relu', padding='same')(up7)
    conv7 = keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv7)
    
    up8 = keras.layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv7)
    up8 = keras.layers.concatenate([up8, conv2])
    conv8 = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(up8)
    conv8 = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv8)
    
    up9 = keras.layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv8)
    up9 = keras.layers.concatenate([up9, conv1])
    conv9 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(up9)
    conv9 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv9)
    
    outputs = keras.layers.Conv2D(1, 1, activation='sigmoid')(conv9)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def load_images_masks(image_paths, mask_paths):
    images = []
    masks = []
    
    for image_path, mask_path in zip(image_paths, mask_paths):
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Preprocess image and mask if necessary
        # For example, you can resize them to a specific size or normalize pixel values
        
        # Append processed image and mask to the lists
        images.append(image)
        masks.append(mask)
    
    return images, masks

def preprocess(images, masks):
    # Preprocess images and masks here if necessary
    # For example, you can resize images and masks to a specific size or normalize pixel values
    # Ensure that the preprocessing steps are consistent for both images and masks
    
    preprocessed_images = []
    preprocessed_masks = []
    
    for image, mask in zip(images, masks):
        # Perform preprocessing steps on image and mask
        
        # Append preprocessed image and mask to the lists
        preprocessed_images.append(image)
        preprocessed_masks.append(mask)
    
    return preprocessed_images, preprocessed_masks

# Set the path to the folder containing the images and masks
data_path = "/Users/MayraBerrones/Documents/DDSM-CIBIS/Full_enhance"

# Get the list of image and mask file paths
image_paths = [os.path.join(data_path, filename) for filename in os.listdir(data_path) if "FULL" in filename]
mask_paths = [os.path.join(data_path, filename) for filename in os.listdir(data_path) if "MASK" in filename]

# Split the image and mask paths into training and validation sets
train_image_paths, val_image_paths, train_mask_paths, val_mask_paths = train_test_split(image_paths, mask_paths, test_size=0.2, random_state=42)

# Load and preprocess training images and masks
train_images, train_masks = load_images_masks(train_image_paths, train_mask_paths)
train_images, train_masks = preprocess(train_images, train_masks)

# Load and preprocess validation images and masks
val_images, val_masks = load_images_masks(val_image_paths, val_mask_paths)
val_images, val_masks = preprocess(val_images, val_masks)

# Convert images and masks to NumPy arrays
train_images = np.array(train_images)
train_masks = np.array(train_masks)
val_images = np.array(val_images)
val_masks = np.array(val_masks)

# Normalize pixel values of images to the range [0, 1]
train_images = train_images / 255.0
val_images = val_images / 255.0

# Build the U-Net model
input_shape = train_images[0].shape
model = unet(input_shape)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_masks, validation_data=(val_images, val_masks), epochs=10, batch_size=16)

# Save the trained model
model.save("unet_model.h5")
