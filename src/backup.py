#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"
import time

start = time.time()
from reinforce_ import ReinforceFeatureEngineer

# data = pd.read_csv('../data/pathmate.csv', header=None)
# X = data.iloc[:, :-1].values
# y = data.iloc[:, -1].values

from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)
feat = ReinforceFeatureEngineer(pop_size=10, generation=10, use_rl=True)  # target_type='regression'
feat.fit(X, y)
pipeline = feat.predict()

print()