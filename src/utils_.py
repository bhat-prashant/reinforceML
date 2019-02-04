#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"

import numpy as np


def reshape_data(individual):
    individual.data = reshape_numpy(individual.data)


def reshape_numpy(ndarray):
    if ndarray.ndim == 1:
        ndarray = np.reshape(ndarray, (ndarray.shape[0], 1))
    return ndarray












