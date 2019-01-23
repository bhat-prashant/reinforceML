import numpy as np
import random
from sklearn.datasets import load_breast_cancer
from tpot import TPOTClassifier
from deap import creator
from reinforce import FeatureEngineer


from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = data.data
y = data.target

feat = FeatureEngineer(pop_size=100)
feat.fit(X, y)


