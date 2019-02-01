#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"



import importlib
import numpy as np
import logging
from sklearn.model_selection import train_test_split




def sampling_lookup():
    sampling_tq = {
        # under sampling
        'near_miss': 'imblearn.under_sampling.NearMiss',
        'tomek_links': 'imblearn.under_sampling.TomekLinks',
        # over sampling
        'smote': 'imblearn.over_sampling.SMOTE',
        'smote_nc': 'imblearn.over_sampling.SMOTENC',
        # combination of under and over sampling
        'smote_tomek': 'imblearn.combine.SMOTETomek',
        'smote_enn': 'imblearn.combine.SMOTEENN'
    }
    return sampling_tq


# usage : NearMiss = Sampler('near_miss').sampler()
class Sampler:

    # Choose from the list samplers available under sampling lookup
    def __init__(self, technique=None):
        self.technique = technique
        self.strategy = None

    def sample(self):
        if self.technique is not None:
            samplers = sampling_lookup()
            try:
                self.strategy = samplers.get(self.technique)
                self.strategy = self.strategy.rsplit('.', 1)
                mod = importlib.import_module(self.strategy[0], self.strategy[1])
                self.sampler = getattr(mod, self.strategy[1])
                return self.sample()
            except KeyError:
                logging.error('Unknown sampling technique. Going ahead without sampling !')
            except ImportError:
                logging.error('Unable to import sampler class. Going ahead without sampling !')



# Class for subsampling big data set
class SubSampler:
    # subsample big dataset for faster computation
    # default rate is 0.25 , could be changed by setting BaseFeatureEngineer.subsample
    # to a float between 0 and 1 depending on the dataset size

    def __init__(self, size=0.2):
        self.size = size

    def sample(self, X_t, y_t):
        X, _, y, _ = train_test_split(X_t, y_t, train_size=self.size, test_size=0)
        return X, y


