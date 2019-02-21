#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"


import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import  DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from utils_ import reshape_numpy
from sklearn.model_selection import train_test_split



# Test accuracy , feature importance
# Future Work: Feature selection - covariance, varianceThreshold, selectFromModel etc. (sklearn.feature_selection)
def fitness_score(X, y, estimator=RandomForestClassifier(n_estimators=500, random_state=10, n_jobs=-1), score=roc_auc_score):
    X = reshape_numpy(X)
    y = reshape_numpy(y)
    X_train,  X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
    estimator = DecisionTreeClassifier(random_state=10)
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_val)
    reward = score(y_val, y_pred)
    # print('Accuracy score: ', reward)
    return reward, estimator.feature_importances_.tolist()



