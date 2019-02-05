#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"


import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



# Test accuracy , feature importance
# Future Work: Feature selection - covariance, varianceThreshold, selectFromModel etc. (sklearn.feature_selection)
def fitness_score(X, y, estimator=GradientBoostingClassifier(n_estimators=1, random_state=10),
                  score=accuracy_score):
    if X.ndim == 1:
        np.reshape(X, (X.shape[0], 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    reward = score(y_test, y_pred)
    return reward, estimator.feature_importances_.tolist()



