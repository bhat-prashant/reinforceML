'''
TODO --
1. Linear model for assessing quality of dataset and quality of each chromosome / feature
2. RandomForest with 500 estimators for initial and final evaluation
3.
'''
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# test accuracy
def final_reward(X, y, estimator=RandomForestClassifier(n_estimators=500), score=accuracy_score, coef=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    if hasattr(estimator, 'coef_') and coef:
        return score(y_test, y_pred), estimator.coef_
    return score(y_test, y_pred)
