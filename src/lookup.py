#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler, MaxAbsScaler, MinMaxScaler, Normalizer, \
    PolynomialFeatures, PowerTransformer, QuantileTransformer, RobustScaler
from sklearn.feature_selection import SelectKBest, SelectFromModel, SelectPercentile, SelectFpr, SelectFdr, \
    VarianceThreshold, SelectFwe, RFE, RFECV, chi2, f_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC

from transformer import AddReinforce, SubtractReinforce, KBinsDiscretizerReinforce, EmptyTransformer


class TransformerLookUp:
    def __init__(self, X_col):
        self._X_col = X_col
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

    def _create_transformers(self):
        self._unary_transformers = {
            # 'KBinsDiscretizer': {
            #     'transformer': KBinsDiscretizerReinforce,
            #     'params': {
            #         'strategy': ['uniform', 'quantile'],
            #         'n_bins': [3, 5, 7, 10, 15, 20, 30],
            #         'index0': np.arange(0, self._X_col, 1)}
            # },
            # 'AddNumpy': {
            #     'transformer': AddReinforce,
            #     'params': {'index0': np.arange(0, self._X_col, 1),
            #                'index1': np.arange(0, self._X_col, 1)}
            # },
            # 'SubtractNumpy': {
            #     'transformer': SubtractReinforce,
            #     'params': {'index0': np.arange(0, self._X_col, 1),
            #                'index1': np.arange(0, self._X_col, 1)}
            # },
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
                'params': {'feature_range': [(0,1), (-1,1)]}
            },
            'Normalizer': {
                'transformer': Normalizer,
                'params': {'norm': ['l1', 'l2', 'max']}
            },
            'PowerTransformer': {
                'transformer': PowerTransformer,
                'params': {'method': ['yeo-johnson', 'box-cox'],
                           'standardize': [True, False]}
            },
            'PolynomialFeatures': {
                'transformer': PolynomialFeatures,
                'params': {'degree': [2,3,4],
                           'include_bias': [True, False],
                           'interaction_only': [True, False]}
            },
            'QuantileTransformer': {
                'transformer': QuantileTransformer,
                'params': {'n_quantiles': [2, 3, 4],
                           'output_distribution': ['uniform', 'normal']}
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
                'params': {'estimator': [RandomForestClassifier(n_estimators=10, random_state=10),
                                         GradientBoostingClassifier(n_estimators=10, random_state=10)],
                           'threshold': ['mean', 'median', '0.5*mean']}
            },
    
            'SelectPercentile': {
                'transformer': SelectPercentile,
                'params': {'score_func': [chi2, f_classif],
                           'percentile': np.arange(int(self._X_col / 2), self._X_col - 1, 1)}
            },
            'RFE': {
                'transformer': RFE,
                'params': {'estimator': [RandomForestClassifier(n_estimators=10, random_state=10),
                                         GradientBoostingClassifier(n_estimators=10, random_state=10)],
                           'n_features_to_select': np.arange(int(self._X_col / 2), self._X_col - 1, 1)}
            },

            # bit slower than expected. Could be commented out to speed up operations
            'RFECV': {
                'transformer': RFECV,
                'params': {'estimator': [RandomForestClassifier(n_estimators=10, random_state=10),
                                         GradientBoostingClassifier(n_estimators=10, random_state=10)],
                           'min_features_to_select': np.arange(int(self._X_col / 2), self._X_col - 1, 1),
                           'cv': [3, 5, 7]}
            },

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
                'transformer': PCA,
                'params': {'n_components': np.arange(int(self._X_col / 2), self._X_col, 1),
                           'whiten': [True, False],
                           'svd_solver': ['auto', 'full', 'arpack', 'randomized']}
            },
            'EmptyExtractor': {
                'transformer': EmptyTransformer,
                'params': {}
            }
        }

