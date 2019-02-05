#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"
import time
start = time.time()

from sklearn.datasets import load_wine
data = load_wine()
X = data.data
y = data.target

# from reinforce_ import FeatureEngineer
# feat = FeatureEngineer()
# feat.fit(X, y)
#


from tpot import TPOTClassifier

tpo = TPOTClassifier(population_size=100, generations=3)
tpo.fit(X, y)
