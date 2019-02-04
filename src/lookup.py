#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler


def empty_transformer(*args):
    pass


# transformer lookup. Must contain all required parameters
# Future Work : Dynamically import required classes using 'importlib.import_module'
def get_lookup(X_col):
    # X_col - number of features / columns in the original dataset
    trans_lookup = {
        'EmptyTransformer': {'source': empty_transformer,
                             'root': True,
                             'params': {}
                             },
        'KBinsDiscretizer': {'source': KBinsDiscretizer,
                             'root': False,
                             'params': {
                                 'KBinsDiscretizer_strategy': ['uniform', 'quantile', 'kmeans'],
                                 'KBinsDiscretizer_n_bins': [3, 5, 7, 10, 15, 20, 30]}
                             },
        'StandardScaler': {'source': StandardScaler,
                           'root': True,
                           'params': {'StandardScaler_with_mean': [True, False],
                                      'StandardScaler_with_std': [True, False]}
                           },
        'PCA': {'source': PCA,
                'root': True,
                'params': {'PCA_n_components': np.arange(1, X_col, 1),
                           'PCA_whiten': [True, False],
                           'PCA_svd_solver': ['auto', 'full', 'arpack', 'randomized']}
                },
        'Add': {'source': np.add,
                'root': False,
                'params': {'Add_index0': np.arange(0, X_col, 1),
                           'Add_index1': np.arange(0, X_col, 1)}
                },
        'Subtract': {'source': np.add,
                     'root': False,
                     'params': {'Subtract_index0': np.arange(0, X_col, 1),
                                'Subtract_index1': np.arange(0, X_col, 1)}
                     },

    }

    return trans_lookup
