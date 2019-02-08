#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"
import time

start = time.time()

from sklearn.datasets import load_digits

data = load_digits()
X = data.data
y = data.target


from reinforce_ import FeatureEngineer

feat = FeatureEngineer(pop_size=10)
pipeline = feat.fit(X, y)
X = pipeline.fit_transform(X, y)

from metrics_ import fitness_score
score, _ = fitness_score(X, y)
print(score)


