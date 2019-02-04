#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"
import time

start = time.time()

from sklearn.datasets import load_wine

data = load_wine()
X = data.data
y = data.target

from reinforce_ import FeatureEngineer

feat = FeatureEngineer(pop_size=100)
feat.fit(X, y)


