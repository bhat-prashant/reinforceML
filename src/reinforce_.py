#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"

import warnings

from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
from base_ import BaseReinforceML
from sklearn.metrics import accuracy_score, r2_score
from sklearn.svm import SVC, SVR
from transformer import ExtractedArray, SelectedArray
import numpy as np

random_state = np.random.RandomState(10)


class ReinforceFeatureEngineer(BaseReinforceML):
    def __init__(self, generation=20, pop_size=100, use_rl=True, mutation_rate=0.3, crossover_rate=0.7,
                 target_type='classification', scorer=accuracy_score, trans_types=None,
                 estimator=SVC(random_state=random_state, gamma='auto'), rl_technique='ddqn'):
        """ Automated Feature Engineer (AFE)

        Generates best set of features given a target type (classification / regression)

        :param generation: number of generations to run during evolution, default to 20.
        :param pop_size: number of different individuals / pipelines to be created during evolution (population size)
        :param mutation_rate: rate of mutation during evolution
        :param crossover_rate: rate of cross over during evolution. sum of mutation_rate and crossover_rate should sum to 1
        :param target_type: one of 'classification', 'regression'
                            target type defines type of fitness estimator. Defaults to 'classification'
        :param scorer: one of sklearn metrics, usually one of accuracy_score / r2_score depending on the target_type
                       can be left to default in case you mention target_type
        :param estimator: an instance of sklearn estimator depending on target_type
                          can be left to default in case you mention target_type
        :return: None
        """

        if target_type == 'regression':
            estimator = SVR(gamma='auto')
            scorer = r2_score
        if trans_types is None:
            trans_types = ['unary', 'scaler', 'selector', 'extractor']

        super(ReinforceFeatureEngineer, self).__init__(estimator=estimator,
                                                       feateng=True,
                                                       generation=generation, pop_size=pop_size,
                                                       mutation_rate=mutation_rate,
                                                       crossover_rate=crossover_rate,
                                                       scorer=scorer, inputArray=[np.ndarray],
                                                       outputArray=ExtractedArray,
                                                       trans_types=trans_types,
                                                       random_state=random_state,
                                                       use_rl=use_rl, rl_technique=rl_technique)

    def predict(self, X=None, y=None):
        """ Returns a pipeline that yields the best score for the given estimator and scorer

        :param X: numpy ndarray input matrix [n_samples, n_features]
        :param y: numpy ndarray target values
        :return: a tuple of sklearn pipeline and a instance of estimator
        """
        self._estimator.fit(self._X_train, self._y_train)
        y_pred = self._estimator.predict(self._X_val)
        self._initial_score = self._scorer(self._y_val, y_pred)
        print('Initial Best score : ', self._initial_score)
        print(self._hof[0])
        return self._compile_to_sklearn(self._hof[0])


class ReinforceClassifier(BaseReinforceML):
    def __init__(self, generation=20, pop_size=100, mutation_rate=0.3, use_rl=True, crossover_rate=0.7,
                 trans_types=None,
                 scorer=accuracy_score, estimator=SVC(random_state=random_state, gamma='auto'), rl_technique='ddqn'):
        """ Automated Classification

        Given a labeled inputs, searches for an optimal pipeline which maximises the scorer(accuracy_score by default)

        :param generation: number of generations to run during evolution, default to 20.
        :param pop_size: number of different individuals / pipelines to be created during evolution (population size)
        :param mutation_rate: rate of mutation during evolution
        :param crossover_rate: rate of cross over during evolution. sum of mutation_rate and crossover_rate should sum to 1
        :param scorer: one of sklearn metrics, usually one of accuracy_score
        :return: None
        """

        if trans_types is None:
            trans_types = ['unary', 'scaler', 'selector', 'extractor', 'classifier']

        super(ReinforceClassifier, self).__init__(estimator=estimator,
                                                  feateng=False,
                                                  generation=generation, pop_size=pop_size, mutation_rate=mutation_rate,
                                                  crossover_rate=crossover_rate,
                                                  scorer=scorer, inputArray=[np.ndarray], outputArray=SelectedArray,
                                                  trans_types=trans_types,
                                                  random_state=random_state, use_rl=use_rl, rl_technique=rl_technique)

    def predict(self, X=None, y=None):
        """ Returns a pipeline that yields the best score for the given estimator and scorer

        :param X: numpy ndarray input matrix [n_samples, n_features]
        :param y: numpy ndarray target values
        :return: a tuple of sklearn pipeline and a instance of estimator
        """
        print(self._hof[0])
        return self._compile_to_sklearn(self._hof[0])
