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
        'EmptyTransformer': {'package' : 'other',
                             'transformer': empty_transformer,
                             'root': True,
                             'params': {}
                             },
        'KBinsDiscretizer': {'package' : 'sklearn',
                             'transformer': KBinsDiscretizer,
                             'root': False,
                             'params': {
                                 'strategy': ['uniform', 'quantile', 'kmeans'],
                                 'n_bins': [3, 5, 7, 10, 15, 20, 30],
                                 'index0': np.arange(0, X_col, 1)}
                             },
        'StandardScaler': {'package' : 'sklearn',
                           'transformer': StandardScaler,
                           'root': True,
                           'params': {'with_mean': [True, False],
                                      'with_std': [True, False]}
                           },
        'PCA': {'package' : 'sklearn',
                'transformer': PCA,
                'root': True,
                'params': {'n_components': np.arange(1, X_col, 1),
                           'whiten': [True, False],
                           'svd_solver': ['auto', 'full', 'arpack', 'randomized']}
                },
        'Add': {'package' : 'numpy',
                'transformer': np.add,
                'root': False,
                'params': {'index0': np.arange(0, X_col, 1),
                           'index1': np.arange(0, X_col, 1)}
                },
        'Subtract': {'package' : 'numpy',
                     'transformer': np.subtract,
                     'root': False,
                     'params': {'index0': np.arange(0, X_col, 1),
                                'index1': np.arange(0, X_col, 1)}
                     },
        'Multiply': {'package': 'numpy',
                     'transformer': np.multiply,
                     'root': False,
                     'params': {'index0': np.arange(0, X_col, 1),
                                'index1': np.arange(0, X_col, 1)}
                     },
        'Divide': {'package': 'numpy',
                     'transformer': np.divide,
                     'root': False,
                     'params': {'index0': np.arange(0, X_col, 1),
                                'index1': np.arange(0, X_col, 1)}
                     },
        'Log': {'package': 'numpy',
                   'transformer': np.log,
                   'root': False,
                   'params': {'index0': np.arange(0, X_col, 1) }
                   },

    }

    return trans_lookup
