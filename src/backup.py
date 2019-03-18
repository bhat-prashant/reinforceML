#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"
import time
import pandas as pd
start = time.time()
from sklearn.model_selection import train_test_split
from reinforce_ import FeatureEngineer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.datasets import load_boston, load_diabetes
# data  = pd.read_csv('../data/pathmate.csv', header=None)
# X = data.iloc[:, :-1].values
# y = data.iloc[:, -1].values

X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

feat = FeatureEngineer(pop_size=10, generation=5, target_type='regression')
feat.fit(X, y)
pipeline, estimator = feat.predict()

# # original dataset
# estimator.fit(X_train, y_train)
# y_pred = estimator.predict(X_test)
# initauc = roc_auc_score(y_test, y_pred)
#
# # transformed  dataset
# pipeline.fit(X_train, y_train)
# y_pred_t = pipeline.predict(X_test)
# finalauc= roc_auc_score(y_test, y_pred_t)
# print('initial auc {} and final auc {}'.format(initauc, finalauc))
# error = ((1-initauc)-(1-finalauc)) / (1-initauc)
# print('Error Reduction: ', error*100)


