#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"

from sklearn.model_selection import train_test_split
from reinforce_ import ReinforceClassifier
from sklearn.datasets import load_digits

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

feat = ReinforceClassifier(pop_size=10, generation=5,)# target_type='regression'
feat.fit(X_train, y_train)
pipeline = feat.predict()
