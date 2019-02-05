#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"

import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.decomposition import PCA


def reshape_data(individual):
    individual.data = reshape_numpy(individual.data)


def reshape_numpy(ndarray):
    if ndarray.ndim == 1:
        ndarray = np.reshape(ndarray, (ndarray.shape[0], 1))
    return ndarray

def operator_precheck(operator, input_matrix, **kwargs):
    if isinstance(operator, PCA):
        if kwargs['n_components'] > input_matrix.shape[1]:
            kwargs['n_components'] = input_matrix.shape[1] - 1
            operator = PCA(**kwargs)
        return operator, input_matrix

    return operator, input_matrix







