
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
from pyimagesearch import config

import numpy as np
import pickle
import os

def load_data_split(splitPath):
    data = []
    labels = []
    for row in open(splitPath):
        row = row.strip().split(",")
        label = row[0]
        features = np.array(row[1:], dtype = "float")
        data.append(features)
        labels.append(label)
    data = np.array(data)
    labels = np.array(labels)
    return (data, labels)

testingPath = os.path.sep.join([config.BASE_CSV_PATH,
	"{}.csv".format(config.VAL)])
# load the data from disk

print("[INFO] loading data...")

(testX, testY) = load_data_split(testingPath)
# load the label encoder from disk
le = pickle.loads(open(config.LE_PATH, "rb").read())

# train the model
print("[INFO] LOADING model...")

with open(config.MODEL_PATH, "rb") as model:
    # evaluate the model
    print("[INFO] evaluating...")
    preds = model.predict(testX)
    print(classification_report(testY, preds, target_names=le.classes_))




