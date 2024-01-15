import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout, Lambda
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
import cv2
import glob


# Paths to data
input_images_path = "/Users/MayraBerrones/Documents/DDSM-CIBIS/Training/Full_images"
input_masks_path = "/Users/MayraBerrones/Documents/DDSM-CIBIS/Training/All_masks"

def combine_masks(mask_folder):
    # Get list of all mask images in the folder
    mask_files = glob.glob(os.path.join(mask_folder, "*.png"))  # adjust the file type if needed

    # Dictionary to hold combined masks
    combined_masks = {}

    # Iterate over all mask files
    for mask_file in mask_files:
        # Extract base file name, remove the suffix like '_MASK_1', '_MASK_2' etc.
        base_name = "_".join(mask_file.split('_')[:-1])

        # Read the mask image
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)  # or cv2.IMREAD_COLOR if your masks are not grayscale

        # If there is already a mask for this base image, combine (bitwise OR operation)
        if base_name in combined_masks:
            combined_masks[base_name] = cv2.bitwise_or(combined_masks[base_name], mask)
        else:
            combined_masks[base_name] = mask

    # Now you have a dictionary where the key is the base image name and the value is the combined mask image
    # You can write these to disk, return them from a function, etc.
    return combined_masks


# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

# Define input layer
inputs = Input((256, 256, 3))

# Load the pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_tensor=inputs)

s = Lambda(lambda x: x / 255) (inputs)

base_model.trainable = False
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

# Data augmentation
data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

seed = 1

def single_channel_generator(gen):
    for batch in gen:
        # Convert to single channel
        gray_batch = tf.image.rgb_to_grayscale(batch)
        yield gray_batch

image_generator = image_datagen.flow_from_directory(
    input_images_path,
    class_mode=None,
    seed=seed,
    target_size=(256, 256))

mask_generator = single_channel_generator(
    mask_datagen.flow_from_directory(
        input_masks_path,
        class_mode=None,
        seed=seed,
        color_mode='rgb',  # Ensure masks are read in RGB mode initially
        target_size=(256, 256)
    )
)

def combine_generator(gen1, gen2):
    while True:
        yield(next(gen1), next(gen2))

train_generator = combine_generator(image_generator, mask_generator)

model.fit_generator(
    train_generator,
    steps_per_epoch=2000,
    epochs=50)

model.save("unet_model.h5")