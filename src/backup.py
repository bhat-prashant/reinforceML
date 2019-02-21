#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"
import time
import pandas as pd
start = time.time()
from sklearn.model_selection import train_test_split
from reinforce_ import FeatureEngineer
from sklearn.metrics import accuracy_score, roc_auc_score

data  = pd.read_csv('../data/pathmate.csv', header=None)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

feat = FeatureEngineer(pop_size=10, generation=5)
pipeline, estimator = feat.fit(X, y)


# original dataset
estimator.fit(X_train, y_train)
y_pred = estimator.predict(X_test)
initauc = roc_auc_score(y_test, y_pred)

# transformed dataset
pipeline.fit(X_train, y_train)
y_pred_t = pipeline.predict(X_test)
finalauc= roc_auc_score(y_test, y_pred_t)
print('initial auc {} and final auc {}'.format(initauc, finalauc))
error = ((1-initauc)-(1-finalauc)) / (1-initauc)
print('Error Reduction: ', error*100)



######################## IMPORTANT ####################################
'''
1. Build the initial populatiion - 100 individuals
2. collect the data - individuals config vs evaluation metric  (fit_partial)
3. as the evolution progresses, keep building the dataset and update weights of a regression model
4. As the time progresses, regression model becomes clever at suggesting better configs 
5. This can even be used as a lookup metric to save time!
6. After each generation, add some new individuals with possible  highest metric!
7. same thing can be replaced by reinforcement learning here!!


'''