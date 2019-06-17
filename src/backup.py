#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"
import time

start = time.time()
from reinforce_ import ReinforceFeatureEngineer
import pandas as pd

data = pd.read_csv('../data/pathmate.csv', header=None)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

feat = ReinforceFeatureEngineer(pop_size=10, generation=5, )  # target_type='regression'
feat.fit(X, y)
pipeline = feat.predict()

print()