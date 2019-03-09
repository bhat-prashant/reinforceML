#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"
import time
import pandas as pd
from datetime import datetime
import os
import csv

start = time.time()
from sklearn.model_selection import train_test_split
from reinforce_ import FeatureEngineer
from sklearn.metrics import accuracy_score, roc_auc_score

data  = pd.read_csv('../data/diabetes.csv', header=None)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

feat = FeatureEngineer(pop_size=50, generation=10)
pipeline, estimator = feat.fit(X, y)


# original dataset
estimator.fit(X_train, y_train)
y_pred = estimator.predict(X_test)
initauc = roc_auc_score(y_test, y_pred)

# transformed dataset
pipeline.fit(X_train, y_train)
y_pred_t = pipeline.predict(X_test)
finalauc= roc_auc_score(y_test, y_pred_t)
print('initial auc {} and final auc {}'.format(initauc, finalauc))
error = ((1-initauc)-(1-finalauc)) / (1-initauc)
print('Error Reduction: ', error*100)



# Future Work: remove emptyTransformers
row = [datetime.now(), feat._estimator.__class__.__name__, feat._hof[0], initauc, finalauc, "{}%".format(error) ]

with open("../results/results.csv", "a") as results:
    CSVWriter = csv.writer(results)
    CSVWriter.writerow(row)

from sklearn.externals import joblib
filepath = '../models/pathmate.pkl'
if os.path.exists(filepath):
    os.remove(filepath)
joblib.dump(pipeline, filepath, compress=1)
