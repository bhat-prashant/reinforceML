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
    if individual.data.ndim == 1:
        individual.data = np.reshape(individual.data, (individual.data.shape[0], 1))

