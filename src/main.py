#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"
import pandas as pd
import time
start = time.time()
from reinforce_ import FeatureEngineer

from sklearn.datasets import load_wine
data = load_wine()
X = data.data
y = data.target

# dataset = pd.read_csv('poker.csv', header=None)
# X = dataset.iloc[:, :10].values
# y = dataset.iloc[:, 10].values

# Future Work: Regression. For now, only binary classification is supported
# For now, use only numerical inputs and outputs.
# Future Work: Preprocessing is prending such as handling categorical, datetime etc.
feat = FeatureEngineer(pop_size=100, generation=10, random_state=10, append_original=False)
feat.fit(X, y)
feat.transform()


end = time.time()
print("Execution time : ", end - start)











