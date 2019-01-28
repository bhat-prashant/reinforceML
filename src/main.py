#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"

import time
start = time.time()
from reinforce_ import FeatureEngineer

from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = data.data
y = data.target

# Future Work: Regression. For now, only binary classification is supported
# For now, use only numerical inputs and outputs.
# Future Work: Preprocessing is prending such as handling categorical, datetime etc.
feat = FeatureEngineer(pop_size=40, generation=15)
feat.fit(X, y)
feat.transform()

end = time.time()
print("Execution time : ", end - start)


