from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression
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


trainingPath = os.path.sep.join([config.BASE_CSV_PATH,
	"{}.csv".format(config.TRAIN)])
testingPath = os.path.sep.join([config.BASE_CSV_PATH,
	"{}.csv".format(config.TEST)])

validationPath = os.path.sep.join([config.BASE_CSV_PATH,
	"{}.csv".format(config.VAL)])


# load the data from disk
print("[INFO] loading data...")
(trainX, trainY) = load_data_split(trainingPath)
(testX, testY) = load_data_split(testingPath)
(valX, valY) = load_data_split(validationPath)
# load the label encoder from disk
le = pickle.loads(open(config.LE_PATH, "rb").read())

# train the model
print("[INFO] training  histgradient model...")
model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200)

#model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)

#model = HistGradientBoostingClassifier(max_iter=50)


model.fit(trainX, trainY)
# evaluate the model
print("[INFO] evaluating...")
preds = model.predict(testX)
print(classification_report(testY, preds, target_names=le.classes_))

preds_val = model.predict(valX)
print(classification_report(valY, preds_val, target_names=le.classes_))

# serialize the model to disk
print("[INFO] saving model...")
f = open(config.MODEL_PATH, "wb")
f.write(pickle.dumps(model))
f.close()
