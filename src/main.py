# #!/usr/bin/env python
# __author__ = "Prashant Shivarm Bhat"
# __email__ = "PrashantShivaram@outlook.com"
# import time
# start = time.time()
#
# from sklearn.datasets import load_wine
# data = load_wine()
# X = data.data
# y = data.target
#
# # from reinforce_ import FeatureEngineer
# # feat = FeatureEngineer()
# # feat.fit(X, y)
# #
#
#
# from tpot import TPOTClassifier
#
# tpo = TPOTClassifier(population_size=100, generations=3)
# tpo.fit(X, y)

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier



import time
import pandas as pd
start = time.time()
from sklearn.model_selection import train_test_split
from reinforce_ import FeatureEngineer
from sklearn.metrics import accuracy_score, roc_auc_score

data  = pd.read_csv('../data/wind.csv')
X = data.iloc[:, :14].values
y = data.iloc[:, 14].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
# from sklearn.preprocessing import StandardScaler
# scalar = StandardScaler()
# X_train = scalar.fit_transform(X_train)
# X_test = scalar.transform(X_test)
# from sklearn.decomposition import PCA
# pca = PCA(n_components=10)
# X_train = pca.fit_transform(X_train)
# X_test = pca.transform(X_test)

cls = GradientBoostingClassifier(n_estimators=200, random_state=10)
cls.fit(X_train, y_train)
ypred = cls.predict(X_test)
score = roc_auc_score(y_test, ypred)
print(score)




