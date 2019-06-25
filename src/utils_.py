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

def get_individual_config(columns, individual):
    state = dict((el, 0) for el in columns)
    idx = individual.height + 1
    for terminal in individual[idx:]:
        state[terminal.name] = 1
    return np.array(list(state.values()))
