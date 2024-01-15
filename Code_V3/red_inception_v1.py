import os

import tensorflow as tf
from tensorflow import keras
import argparse
from imutils import paths
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

# Import packages

from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation

from tensorflow.keras import backend as K
from tensorflow.keras import utils as np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, Input, Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import  ReduceLROnPlateau

from tensorflow.keras import layers, models, Model, optimizers

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="direccion del dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="Nombre del Plot")
args = vars(ap.parse_args())

INIT_LR = 0
BS = 16
EPOCHS = 20

Hg = 224
Lng = 224

print("[INFO] Cargando imagenes...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (Hg, Lng))
    data.append(image)
    labels.append(label)

data = np.array(data, dtype="float") / 255.0

le = LabelEncoder()
labels = le.fit_transform(labels)
labels = np_utils.to_categorical(labels, 2)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)


train_datagen = ImageDataGenerator(rotation_range=20, 
                            zoom_range=0.15, 
                            width_shift_range=0.2,  
                            height_shift_range=0.2,
                             shear_range=0.15, 
                             horizontal_flip=True,
                             vertical_flip = True,
                             brightness_range=None,
                            zca_whitening=False,
                            zca_epsilon=1e-06,
                             fill_mode="nearest")

def conv_module(input,No_of_filters,filtersizeX,filtersizeY,stride,chanDim,padding="same"):
  input = Conv2D(No_of_filters,(filtersizeX,filtersizeY),strides=stride,padding=padding)(input)
  input = BatchNormalization(axis=chanDim)(input)
  input = Activation("relu")(input)
  return input

def inception_module(input,numK1x1,numK3x3,numk5x5,numPoolProj,chanDim):
                                 #Step 1
  conv_1x1 = conv_module(input, numK1x1, 1, 1,(1, 1), chanDim) 
                                 #Step 2
  conv_3x3 = conv_module(input, numK3x3, 3, 3,(1, 1), chanDim)
  conv_5x5 = conv_module(input, numk5x5, 5, 5,(1, 1), chanDim)
                                 #Step 3
  pool_proj = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input)
  pool_proj = Conv2D(numPoolProj, (1, 1), padding='same', activation='relu')(pool_proj)
                                 #Step 4
  input = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=chanDim)
  return input

def downsample_module(input,No_of_filters,chanDim):
  conv_3x3=conv_module(input,No_of_filters,3,3,(2,2),chanDim,padding="valid")
  pool = MaxPooling2D((3,3),strides=(2,2))(input)
  input = concatenate([conv_3x3,pool],axis=chanDim)
  return input


def MiniGoogleNet(width,height,depth,classes):
  inputShape=(height,width,depth)
  chanDim=-1

  # (Step 1) Define the model input
  inputs = Input(shape=inputShape)

  # First CONV module
  x = conv_module(inputs, 96, 3, 3, (1, 1),chanDim)

  # (Step 2) Two Inception modules followed by a downsample module
  x = inception_module(x, 32, 32,32,32,chanDim)
  x = inception_module(x, 32, 48, 48,32,chanDim)
  x = downsample_module(x, 80, chanDim)
  
  # (Step 3) Five Inception modules followed by a downsample module
  x = inception_module(x, 112, 48, 32, 48,chanDim)
  x = inception_module(x, 96, 64, 32,32,chanDim)
  x = inception_module(x, 80, 80, 32,32,chanDim)
  x = inception_module(x, 48, 96, 32,32,chanDim)
  x = inception_module(x, 112, 48, 32, 48,chanDim)
  x = downsample_module(x, 96, chanDim)

  # (Step 4) Two Inception modules followed
  x = inception_module(x, 176, 160,96,96, chanDim)
  x = inception_module(x, 176, 160, 96,96,chanDim)
  
  # Global POOL and dropout
  x = AveragePooling2D((7, 7))(x)
  x = Dropout(0.5)(x)

  # (Step 5) Softmax classifier
  x = Flatten()(x)
  x = Dense(classes)(x)
  x = Activation("softmax")(x)

  # Create the model
  model = Model(inputs, x, name="googlenet")
  return model

INIT_LR = 5e-3
def poly_decay(epoch):
  maxEpochs = EPOCHS
  baseLR = INIT_LR
  power = 1.0
  alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power
  return alpha



transfer_model = MiniGoogleNet(width=Lng, height=Hg, depth=3, classes=2)


transfer_model.compile(loss="binary_crossentropy", optimizer= "sgd", metrics=["acc"])
H = transfer_model.fit(train_datagen.flow(trainX, trainY, batch_size=BS), 
                                    validation_data=(testX, testY), 
                                    validation_steps = 1000,
                                    steps_per_epoch=1000, 
                                    epochs=EPOCHS)


print("[INFO] Evaluating network...")
predictions = transfer_model.predict(testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=le.classes_))
transfer_model.save('Inc_DIbalance.h5')

# Plot 
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_accuracy")
plt.plot(N, H.history["val_acc"], label="val_accuracy")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

# Rename and move .dcm files.
for (curdir, dirs, files) in os.walk(top=top, topdown=False):

    dirs.sort()
    files.sort()

    print(f"WE ARE AT: {curdir}")
    print("=" * 10)
    
    for f in files:
        
        # === Step 1: Rename .dcm file ===
        if f.endswith(".dcm"):
            
            old_name_path = os.path.join(curdir, f)
            new_name = new_name_dcm(dcm_path=old_name_path)
            
            if new_name:
                new_name_path = os.path.join(curdir, new_name)
                os.rename(old_name_path, new_name_path)
        
                # === Step 2: Move RENAMED .dcm file ===
                move_dcm_up(dest_dir=parent_dir, source_dir=new_name_path, dcm_filename=new_name)
    
    print()
    print("Moving one folder up...")
    print("-" * 40)
    print()
