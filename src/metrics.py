'''
TODO --
1. Linear model for assessing quality of dataset and quality of each chromosome / feature
2. RandomForest with 500 estimators for initial and final evaluation
3.
'''
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import random
import numpy as np

# Test accuracy , feature importance
def fitness_score(X, y, estimator=RandomForestClassifier(n_estimators=100),
                  score=accuracy_score, coef=False, random_state=None):
    if X.ndim == 1:
        np.reshape(X, (X.shape[0], 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    return score(y_test, y_pred), estimator.feature_importances_


# evaluation function for each individual
def evaluate(individual, y):
    X = individual.data
    if X.ndim == 1:
        X = np.reshape(X, (X.shape[0], 1))
    return fitness_score(X, y)

# crossover / mate function for two individuals
def mate(individual1, individual2, relevacne=0.25):
    X1 = individual1.value
    X2 = individual2.value
    i1 = [i for i in range(individual1.feature_importance) if individual1.feature_importance[i] > relevacne]
    X1 = X1[:, i1]
    i2 = [i for i in range(individual1.feature_importance) if individual1.feature_importance[i] > relevacne]
    X2 = X2[:, i2]
    return np.append(X1, X2, axis=1)

# Future Work: Reinforcement Learning
def mutate(individual, transformers):
    X = individual.value
    key = random.choice(list(transformers.keys()))
    # Future Work : Decorators for pre and post sanity check
    X = transformers[key].transform(X)

