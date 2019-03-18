#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"

from sklearn.exceptions import DataConversionWarning
import warnings

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
from main import BaseReinforceML
from utils_ import reshape_numpy,append_to_dataframe
from sklearn.model_selection import train_test_split
from copy import deepcopy
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.svm import SVC, SVR
from transformer import  ExtractedArray
import numpy as np


class FeatureEngineer(BaseReinforceML):
    def __init__(self, generation=20, pop_size=100, mutation_rate=0.3, crossover_rate=0.7,
                 target_type='classification', scorer=accuracy_score, estimator = SVC(random_state=10, gamma='auto')):
        reinforce_learner = RandomForestRegressor(n_jobs=-1, n_estimators=500, random_state=10, warm_start=False)
        if target_type == 'regression':
            estimator = SVR(gamma='auto')
            scorer = r2_score
        super(FeatureEngineer, self).__init__(estimator=estimator, reinforce_learner=reinforce_learner,
                                              generation=generation, pop_size=pop_size, mutation_rate=mutation_rate,
                                              crossover_rate=crossover_rate,
                                              scorer=scorer, inputArray=[np.ndarray], outputArray=ExtractedArray,
                                              trans_types=[ 'unary','scaler','extractor', 'selector' ])#

    def fit(self, X, y):
        # Future Work: checking OneHotEncoding, datetime etc
        self._X = reshape_numpy(X)
        self._y = reshape_numpy(y)
        self._X_train, self._X_val, self._y_train, self._y_val = \
            train_test_split(self._X, self._y, test_size=0.2, random_state=10)
        self._feature_count = self._X.shape[1]
        self._setup_pset()
        self._create_dataframe()
        self._setup_toolbox()
        self._pop = self._toolbox.population(self._pop_size)
        self._evolve()

    def predict(self, X=None, y=None):
        self._estimator.fit(self._X_train, self._y_train)
        y_pred = self._estimator.predict(self._X_val)
        self._initial_score = self._scorer(self._y_val, y_pred)
        print('Initial Best score : ', self._initial_score)
        print(self._hof[0])
        return self._compile_to_sklearn(self._hof[0]), self._estimator

    def _evaluate(self, individual):
        input_matrix = deepcopy(self._X_train)
        target = self._y_train
        pipeline = self._compile_to_sklearn(individual=individual)
        score = 0
        try:
            pipeline.fit(input_matrix, target)
            y_pred = pipeline.predict(self._X_val)
            score = self._scorer(self._y_val, y_pred)
            self._rl_dataframe = append_to_dataframe(self._rl_dataframe, self._pandas_columns, individual, score)
            return score,
        # Future Work: Handle these exceptions
        except Exception as e:
            self._rl_dataframe = append_to_dataframe(self._rl_dataframe, self._pandas_columns, individual, score)
            return score,
