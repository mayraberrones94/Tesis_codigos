from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

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
print("[INFO] training model...")

dtr_model = DecisionTreeClassifier()
rfc_model = RandomForestClassifier()
knn_model = KNeighborsClassifier()
svc_model = SVC(probability = True, random_state = 42)

lr = LogisticRegression()


eclf = StackingClassifier(estimators=[('DTC', dtr_model), ('RFC', rfc_model), 
                        ('KNN', knn_model), ('SVC', svc_model)],
                        final_estimator = lr)


eclf.fit(trainX, trainY)
# evaluate the model
print("[INFO] evaluating...")
preds = eclf.predict(testX)
print(classification_report(testY, preds, target_names=le.classes_))

preds_val = eclf.predict(valX)
print(classification_report(valY, preds_val, target_names=le.classes_))

# serialize the model to disk
print("[INFO] saving model...")
f = open(config.MODEL_PATH, "wb")
f.write(pickle.dumps(eclf))
f.close()