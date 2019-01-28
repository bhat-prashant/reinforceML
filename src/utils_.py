#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"


'''
TODO --
1. Logging
2. Saving results in a ordered fashion to a file
3. creating a file with set of transformations to apply on particular feature. This should be 'readymade' to be consumed by the user

'''
import numpy as np

def reshape_data(individual):
    individual.data = reshape_numpy(individual.data)

def reshape_numpy(ndarray):
    if ndarray.ndim == 1:
        ndarray = np.reshape(ndarray, (ndarray.shape[0], 1))
    return ndarray

