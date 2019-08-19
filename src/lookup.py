#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, SelectFromModel, VarianceThreshold, RFE, RFECV, chi2, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, Normalizer, \
    PolynomialFeatures, QuantileTransformer, RobustScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from transformer import AddReinforce, SubtractReinforce, KBinsDiscretizerReinforce, EmptyTransformer, \
    PCAReinforce, MultiplyReinforce, DivideReinforce, LogReinforce


class TransformerLookUp:
    def __init__(self, X_col, random_state):
        self._X_col = X_col
        self._random_state = random_state
        self._create_transformers()

    # transformer lookup. Must contain all required parameters
    # Future Work : Dynamically import required classes using 'importlib.import_module'
    def get_lookup(self, trans_type):
        # self._X_col - number of features / columns in the original dataset
        if trans_type == 'unary':
            return self._unary_transformers
        elif trans_type == 'scaler':
            return self._universal_scalers
        elif trans_type == 'selector':
            return self._universal_selectors
        elif trans_type == 'extractor':
            return self._universal_extractors
        elif trans_type == 'classifier':
            return self._classifiers

    def _create_transformers(self):
        self._unary_transformers = {
            'KBinsDiscretizer': {
                'transformer': KBinsDiscretizerReinforce,
                'params': {
                    'strategy': ['uniform', 'quantile'],
                    'n_bins': [3, 5, 7, 10, 15, 20, 30],
                    'index0': np.arange(0, self._X_col, 1)}
            },

            'AddReinforce': {
                'transformer': AddReinforce,
                'params': {'index0': np.arange(0, self._X_col, 1),
                           'index1': np.arange(0, self._X_col, 1)}
            },

            'SubtractReinforce': {
                'transformer': SubtractReinforce,
                'params': {'index0': np.arange(0, self._X_col, 1),
                           'index1': np.arange(0, self._X_col, 1)}
            },

            'MultiplyReinforce': {
                'transformer': MultiplyReinforce,
                'params': {'index0': np.arange(0, self._X_col, 1),
                           'index1': np.arange(0, self._X_col, 1)}
            },

            'DivideReinforce': {
                'transformer': DivideReinforce,
                'params': {'index0': np.arange(0, self._X_col, 1),
                           'index1': np.arange(0, self._X_col, 1)}
            },

            'LogReinforce': {
                'transformer': LogReinforce,
                'params': {'index0': np.arange(0, self._X_col, 1)}
            },

            'EmptyUnary': {
                'transformer': EmptyTransformer,
                'params': {}
            }
        }

        self._universal_scalers = {
            'StandardScaler': {
                'transformer': StandardScaler,
                'params': {'with_mean': [True, False],
                           'with_std': [True, False]}
            },
            'MaxAbsScaler': {
                'transformer': MaxAbsScaler,
                'params': {}
            },
            'MinMaxScaler': {
                'transformer': MinMaxScaler,
                'params': {'feature_range': [(0, 1), (-1, 1)]}
            },
            'Normalizer': {
                'transformer': Normalizer,
                'params': {'norm': ['l1', 'l2', 'max']}
            },

            'PolynomialFeatures': {
                'transformer': PolynomialFeatures,
                'params': {'degree': [2],
                           'include_bias': [True, False],
                           'interaction_only': [True, False]}
            },
            'QuantileTransformer': {
                'transformer': QuantileTransformer,
                'params': {'n_quantiles': [2, 3, 4],
                           'output_distribution': ['uniform', 'normal'],
                           'random_state': [self._random_state]}
            },
            'RobustScaler': {
                'transformer': RobustScaler,
                'params': {'with_centering': [True, False],
                           'with_scaling': [True, False]}
            },
            'EmptyScaler': {
                'transformer': EmptyTransformer,
                'params': {}
            }

        }

        self._universal_selectors = {
            'SelectKBest': {
                'transformer': SelectKBest,
                'params': {'score_func': [chi2, f_classif],
                           'k': np.arange(int(self._X_col / 2), self._X_col - 1, 1)}
            },

            'SelectFromModel': {
                'transformer': SelectFromModel,
                'params': {'estimator': [RandomForestClassifier(n_estimators=10, random_state=self._random_state),
                                         GradientBoostingClassifier(n_estimators=10, random_state=self._random_state)],
                           'threshold': ['mean', 'median', '0.5*mean']}
            },

            'RFE': {
                'transformer': RFE,
                'params': {
                    'estimator': [RandomForestClassifier(n_estimators=10, random_state=self._random_state, n_jobs=-1),
                                  GradientBoostingClassifier(n_estimators=10, random_state=self._random_state)],
                           'n_features_to_select': np.arange(int(self._X_col / 2), self._X_col - 1, 1)}

            },

            # bit slower than expected.
            # Future work: manage n_jobs globally
            'RFECV': {
                'transformer': RFECV,
                'params': {
                    'estimator': [RandomForestClassifier(n_estimators=10, random_state=self._random_state, n_jobs=-1),
                                  GradientBoostingClassifier(n_estimators=10, random_state=self._random_state)],
                           'min_features_to_select': np.arange(int(self._X_col / 2), self._X_col - 1, 1),
                           'cv': [3, 5, 7],
                           'n_jobs': [-1]}
            },
            # Future Work: Select the threshold dynamically (i.e use BaseReinforceTransformer to create a wrapper class)
            'VarianceThreshold': {
                'transformer': VarianceThreshold,
                'params': {'threshold': [0.0, 0.1, 0.2, 0.3, 0.4]}
            },

            'EmptySelector': {
                'transformer': EmptyTransformer,
                'params': {}
            }

        }

        self._universal_extractors = {
            'PCA': {
                'transformer': PCAReinforce,
                'params': {'n_components': np.arange(int(self._X_col / 2), self._X_col, 1),
                           'whiten': [True, False],
                           'svd_solver': ['auto', 'full', 'arpack', 'randomized'],
                           'random_state': [self._random_state]}
            },
            'EmptyExtractor': {
                'transformer': EmptyTransformer,
                'params': {}
            }
        }

        self._classifiers = {
            'GaussianNB': {
                'transformer': GaussianNB,
                'params': {}
            },

            # 'XGBClassifier': {
            #     'transformer': XGBClassifier,
            #     'params': {'n_estimators': [100],
            #                'max_depth': range(1, 11),
            #                'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
            #                'subsample': np.arange(0.05, 1.01, 0.05),
            #                'min_child_weight': range(1, 21),
            #                'nthread': [1]}
            # },
            'BernoulliNB': {
                'transformer': BernoulliNB,
                'params': {'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
                           'fit_prior': [True, False]}
            },
            'MultinomialNB': {
                'transformer': MultinomialNB,
                'params': {'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
                           'fit_prior': [True, False]}
            },
            'DecisionTreeClassifier': {
                'transformer': DecisionTreeClassifier,
                'params': {'criterion': ["gini", "entropy"],
                           'max_depth': range(1, 11),
                           'min_samples_split': range(2, 21),
                           'min_samples_leaf': range(1, 21),
                           'random_state': [self._random_state]}
            },
            'ExtraTreesClassifier': {
                'transformer': ExtraTreesClassifier,
                'params': {'n_estimators': [100],
                           'criterion': ["gini", "entropy"],
                           'max_features': np.arange(0.05, 1.01, 0.05),
                           'min_samples_split': range(2, 21),
                           'min_samples_leaf': range(1, 21),
                           'bootstrap': [True, False],
                           'random_state': [self._random_state]}
            },
            'RandomForestClassifier': {
                'transformer': RandomForestClassifier,
                'params': {'n_estimators': [100],
                           'criterion': ["gini", "entropy"],
                           'max_features': np.arange(0.05, 1.01, 0.05),
                           'min_samples_split': range(2, 21),
                           'min_samples_leaf': range(1, 21),
                           'bootstrap': [True, False],
                           'random_state': [self._random_state]}
            },
            'GradientBoostingClassifier': {
                'transformer': GradientBoostingClassifier,
                'params': {'n_estimators': [100],
                           'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
                           'max_depth': range(1, 11),
                           'min_samples_split': range(2, 21),
                           'min_samples_leaf': range(1, 21),
                           'subsample': np.arange(0.05, 1.01, 0.05),
                           'max_features': np.arange(0.05, 1.01, 0.05),
                           'random_state': [self._random_state]}
            },
            'KNeighborsClassifier': {
                'transformer': KNeighborsClassifier,
                'params': {'n_neighbors': range(1, 101),
                           'weights': ["uniform", "distance"],
                           'p': [1, 2]}
            },
            'LinearSVC': {
                'transformer': LinearSVC,
                'params': {'penalty': ["l1", "l2"],
                           'loss': ["hinge", "squared_hinge"],
                           'dual': [True, False],
                           'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                           'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
                           'max_iter': [100000],
                           'random_state': [self._random_state]}
            },
            'LogisticRegression': {
                'transformer': LogisticRegression,
                'params': {'penalty': ["l1", "l2"],
                           'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
                           'dual': [True, False],
                           'solver': ['sag'],
                           'max_iter': [100000],
                           'multi_class': ['auto'],
                           'random_state': [self._random_state]}
            }
        }
