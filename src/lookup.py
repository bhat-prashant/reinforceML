#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"

from sklearn.preprocessing import KBinsDiscretizer, StandardScaler


# transformer lookup. Must contain all required parameters
# Future Work : Dynamically import required classes using 'importlib.import_module'
def get_lookup():
    trans_lookup = {'KBinsDiscretizer': {'source': KBinsDiscretizer,
                                         'root': False,
                                         'params': {'KBinsDiscretizer_strategy': ['uniform', 'quantile', 'kmeans'],
                                                    'KBinsDiscretizer_n_bins': [3, 5, 7, 10, 15, 20, 30]}
                                         },
                    'StandardScaler': {'source': StandardScaler,
                                       'root': True,
                                       'params': {'StandardScaler_with_mean': [True, False],
                                                  'StandardScaler_with_std': [True, False]}
                                       }

                    }

    return trans_lookup
