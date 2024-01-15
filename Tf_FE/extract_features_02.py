
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications import VGG16, imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.preprocessing import LabelEncoder
from pyimagesearch import HDF5DatasetWriter

from imutils import paths
import numpy as np
import progressbar
import argparse
import random
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help = "Path to dataset")
ap.add_argument("-o", "--output", required=True, help="Path to the output file")
ap.add_argument("-b", "--batch-size", type=int, default=32, help="Batchsize of the images")
ap.add_argument("-s", "--buffer-size", type=int, default=1000, help="Size of feature extraction")

args = vars(ap.parse_args())

bs = args["batch_size"]

print("[Info] Loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
random.shuffle(imagePaths)

labels = [p.split(os.path.sep)[-2] for p in imagePaths]
le = LabelEncoder()
labels = le.fit_transform(labels)

print("[Info] Loading network...")
model = VGG16(weights = "imagenet", include_top = False)
dataset = HDF5DatasetWriter((len(imagePaths), 512 * 7 * 7), 
        args["output"], dataKey = "features", bufSize = args["buffer_size"])
dataset.storeClassLabels(le.classes_)

widgets = ["Extracting Features: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(imagePaths), widgets=widgets).start()

for i in np.arange(0, len(imagePaths), bs):
    batchPaths = imagePaths[i:i + bs]
    batchLabels = labels[i:i + bs]
    batchImages = []

    for (j, imagePath) in enumerate(batchPaths):
        image = load_img(imagePath, target_size = (224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis = 0)
        image = imagenet_utils.preprocess_input(image)
        batchImages.append(image)
    
    batchImages = np.vstack(batchImages)
    features = model.predict(batchImages, batch_size = bs)
    features = features.reshape((features.shape[0], 512 * 7 * 7))
    dataset.add(features, batchLabels)
    pbar.update(i)

dataset.close()
pbar.finish()
