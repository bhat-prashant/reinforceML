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

from transformer import AddNumpy, SubtractNumpy


class TransformerLookUp:
    X_col = 0

    @classmethod
    def set_feature_count(cls, feat_count):
        X_col = feat_count

    # Does nothing!
    @classmethod
    def empty_primitive(cls):
        pass

    # transformer lookup. Must contain all required parameters
    # Future Work : Dynamically import required classes using 'importlib.import_module'
    @classmethod
    def get_lookup(cls, feat_count, trans_type):
        # X_col - number of features / columns in the original dataset
        cls.set_feature_count(feat_count)
        if trans_type == 'unary':
            return cls.unary_transformers
        elif trans_type == 'scaler':
            return cls.universal_scalers
        elif trans_type == 'selector':
            return cls.universal_selectors
        elif trans_type == 'extractor':
            return cls.universal_extractors

    unary_transformers = {
        'KBinsDiscretizer': {
            'transformer': KBinsDiscretizer,
            'params': {
                'strategy': ['uniform', 'quantile', 'kmeans'],
                'n_bins': [3, 5, 7, 10, 15, 20, 30],
                'index0': np.arange(0, X_col, 1)}
        },
        'AddNumpy': {
            'transformer': AddNumpy,
            'params': {'index0': np.arange(0, X_col, 1),
                       'index1': np.arange(0, X_col, 1)}
        },
        'SubtractNumpy': {
            'transformer': SubtractNumpy,
            'params': {'index0': np.arange(0, X_col, 1),
                       'index1': np.arange(0, X_col, 1)}
        }
    }

    universal_scalers = {
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
                       'output_distribution': ['uniform', 'normal'],
                       'interaction_only': [True, False]}
        },
        'RobustScaler': {
            'transformer': RobustScaler,
            'params': {'with_centering': [True, False],
                       'with_scaling': [True, False]}
        },

    }

    universal_selectors = {
        'SelectKBest': {
            'transformer': SelectKBest,
            'params': {'score_func': [chi2, f_classif],
                       'k': np.arange(int(X_col / 2), X_col - 1, 1)}
        },

        'SelectFromModel': {
            'transformer': SelectFromModel,
            'params': {'estimator': [RandomForestClassifier(n_estimators=10, random_state=10),
                                     GradientBoostingClassifier(n_estimators=10, random_state=10),
                                     LinearSVC(random_state=10)],
                       'threshold': ['mean', 'median', '0.5*mean']}
        },

        'SelectPercentile': {
            'transformer': SelectPercentile,
            'params': {'score_func': [chi2, f_classif],
                       'percentile': np.arange(int(X_col / 2), X_col - 1, 1)}
        },

        'SelectFpr': {
            'transformer': SelectFpr,
            'params': {'score_func': [chi2, f_classif]}
        },

        'SelectFdr': {
            'transformer': SelectFdr,
            'params': {'score_func': [chi2, f_classif]}
        },

        'SelectFwe': {
            'transformer': SelectFwe,
            'params': {'score_func': [chi2, f_classif]}
        },

        'RFE': {
            'transformer': RFE,
            'params': {'estimator': [RandomForestClassifier(n_estimators=10, random_state=10),
                                     GradientBoostingClassifier(n_estimators=10, random_state=10),
                                     LinearSVC(random_state=10)],
                       'n_features_to_select': np.arange(int(X_col / 2), X_col - 1, 1)}
        },

        'RFECV': {
            'transformer': RFECV,
            'params': {'estimator': [RandomForestClassifier(n_estimators=10, random_state=10),
                                     GradientBoostingClassifier(n_estimators=10, random_state=10),
                                     LinearSVC(random_state=10)],
                       'min_features_to_select': np.arange(int(X_col / 2), X_col - 1, 1),
                       'cv': [3, 5, 7]}
        },

        'VarianceThreshold': {
            'transformer': VarianceThreshold,
            'params': {'threshold': [0.0, 0.1, 0.2, 0.3, 0.4]}
        },

    }

    universal_extractors = {
        'PCA': {
            'transformer': PCA,
            'params': {'n_components': np.arange(int(X_col / 2), X_col, 1),
                       'whiten': [True, False],
                       'svd_solver': ['auto', 'full', 'arpack', 'randomized']}
        }
    }
