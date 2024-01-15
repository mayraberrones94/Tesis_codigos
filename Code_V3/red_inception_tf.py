
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
from tensorflow.keras import backend as K
from tensorflow.keras import utils as np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Conv2D, MaxPooling2D, ZeroPadding2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import  ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

from tensorflow.keras import layers, models, Model, optimizers
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="direccion del dataset")

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



from tensorflow.keras.applications.inception_v3 import InceptionV3

inception_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(Lng, Hg, 3))

# Freeze four convolution blocks
for layer in inception_model.layers:
    layer.trainable = False

x = inception_model.output
x = Flatten()(x) # Flatten dimensions to for use in FC layers
x = Dense(512, activation='relu')(x)
x = Dropout(0.2)(x) # Dropout layer to reduce overfitting
x = Dense(2, activation='sigmoid')(x) # Softmax for multiclass
transfer_model = Model(inputs=inception_model.input, outputs=x)

B = 5
errors = []

VALIDATION_ACCURACY = []
VALIDAITON_LOSS = []

for i in range(B):
    X_bootstrap, y_bootstrap = resample(data, labels)
    trainX, testX, trainY, testY = train_test_split(X_bootstrap, y_bootstrap, test_size=0.25, random_state=42)
    
    transfer_model.compile(loss="binary_crossentropy", optimizer= "sgd", metrics=["acc"])
    history = transfer_model.fit_generator(train_datagen.flow(trainX, trainY, batch_size=BS), 
                                    validation_data=(testX, testY), 
                                    validation_steps = 1000,
                                    steps_per_epoch=1000, 
                                    epochs=EPOCHS)

    predictions = transfer_model.predict(testX, batch_size=BS)
    print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=le.classes_))
    transfer_model.save("IN_tf_bootstrap_DIBalance"+str(i)+".h5")
    transfer_model.load_weights("IN_tf_bootstrap_DIBalance"+str(i)+".h5")

    N = np.arange(0, EPOCHS)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, history.history["loss"], label="train_loss")
    plt.plot(N, history.history["val_loss"], label="val_loss")
    plt.plot(N, history.history["acc"], label="train_accuracy")
    plt.plot(N, history.history["val_acc"], label="val_accuracy")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="upper right")
    plt.savefig("IN_tf_bootstrap_DIBalance" +str(i)+ ".png")	
	
    results = transfer_model.evaluate(testX, testY)
    results = dict(zip(transfer_model.metrics_names,results))
	
    VALIDATION_ACCURACY.append(results["acc"])
    VALIDAITON_LOSS.append(results["loss"])