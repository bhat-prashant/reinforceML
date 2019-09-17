#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"

import datetime
import pickle

import numpy as np
import pandas as pd

now = datetime.datetime.now()


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


def save_logbook(logbook):
    df_log = pd.DataFrame(logbook)
    df_log.to_csv('../logs/{}.csv'.format(now.strftime("%Y-%m-%d_%H:%M")), index=False)


def save_model(pipelines):
    if isinstance(pipelines, list):
        for num, model in enumerate(pipelines):
            pickle.dump(pipelines, open('../saved_models/{}__{}'.format(now.strftime("%Y-%m-%d_%H:%M"), num), 'wb'))
    else:
        pickle.dump(pipelines, open('../saved_models/{}'.format(now.strftime("%Y-%m-%d_%H:%M")), 'wb'))
