#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"

# Future Work:
# Do NOT apply same transformation again and again !!
# check number of rows in a column. if they are not equal to original number of rows, discard
import numpy as np
from constants import *
import networkx as nx

from utils_ import reshape_data

def unary_decorator(func):
    def unary_check(*args, **kwargs):
        index = kwargs.get(INX)
        trans = args[0]
        individual = kwargs.get(IND)
        check_pass = False

        # Pre-check
        if trans.name == MATH_OPS:
            if not np.any(individual.data[:, index] <= 0):
                check_pass = check_unary_redundancy(individual, index, MATH_OPS)

        if trans.name == KBD:
            check_pass = check_unary_redundancy(individual, index, KBD)

        if check_pass:
            # apply transformation if check_pass is True
            data = func(*args, **kwargs)

            # Post-processing
            if trans.name == SQR or trans.name == SQRT or trans.name == LOG:
                if not np.isnan(data).any() and not np.isinf(data).any():
                    individual.data[:, index] = data[:, 0]
            if trans.name == KBD:
                individual.data[:, index] = data.indices


            reshape_data(individual)

    return unary_check

def universal_decorator(func):
    def universal_check(*args, **kwargs):
        individual = kwargs.get(IND)
        # This condition checks whether universal transformation has been applied before.
        if individual.transformation_graph is None:
            check_pass = True
        else:
            check_pass = check_universal_redundancy(individual, SCALER)

        if check_pass:
            # apply transformation if check_pass is True
            data = func(*args, **kwargs)
            return data

    return universal_check


def check_universal_redundancy(individual, trans_array):
    nodes = list(individual.transformation_graph)
    for node in nodes:
        if [trans for trans in trans_array if trans in node]:
            return False
    return True


def check_unary_redundancy(individual, index, trans_array):
    nodes = list(individual.meta_data[index][A_GRAPH])
    for node in nodes:
        if [trans for trans in trans_array if trans in node]:
            return False
    return True