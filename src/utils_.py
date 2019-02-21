#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"

import numpy as np
import pandas as pd


def reshape_data(individual):
    individual.data = reshape_numpy(individual.data)


def reshape_numpy(ndarray):
    if ndarray.ndim == 1:
        ndarray = np.reshape(ndarray, (ndarray.shape[0], 1))
    return ndarray


# Append individual's transformers to pandas dataframe, later user for RL training
def append_to_dataframe(dataframe, columns, individual, score):
    row = dict((el,0) for el in columns)
    idx = individual.height + 2
    for terminal in individual[idx:]:
        row[terminal.name] = 1
    row[columns[-1]] = score # update reward
    df_row = pd.DataFrame.from_dict([row], orient='columns')
    return pd.concat([dataframe, df_row], sort=False)




