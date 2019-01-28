#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"

# Future Work:
# Do NOT apply same transformation again and again !!
# check number of rows in a column. if they are not equal to original number of rows, discard
import numpy as np
from constants import *
from utils_ import reshape_data

def unary_decorator(func):
    def pre_post_check(*args, **kwargs):
        index = kwargs.get(INX)
        trans = args[0]
        individual = kwargs.get(IND)
        check_pass = False

        # Pre-check
        if trans.name == LOG:
            if not np.any(individual.data[:, index] <= 0):
                check_pass = check_redundancy(individual, index, MATH_OPS)

        if trans.name == SQR:
            check_pass = check_redundancy(individual, index, MATH_OPS)

        if trans.name == SQRT:
            if not np.any(individual.data[:, index] < 0):
                check_pass = check_redundancy(individual, index, MATH_OPS)

        if trans.name in PRE_PROCESSORS:
            check_pass = check_redundancy(individual, index, PRE_PROCESSORS)

        if check_pass:
            data = func(*args, **kwargs)

            # Post-processing
            if trans.name == SQR or trans.name == SQRT or trans.name == LOG:
                if not np.isnan(data).any() and not np.isinf(data).any():
                    individual.data[:, index] = data[:, 0]
            if trans.name == KBD:
                individual.data[:, index] = data.indices
            if trans.name == MMS or trans.name == MAS:
                individual.data[:, index] = data[:, 0]

            reshape_data(individual)

    return pre_post_check


def check_redundancy(individual, index, trans_array):
    nodes = list(individual.meta_data[index][A_GRAPH])
    for node in nodes:
        if [trans for trans in trans_array if trans in node]:
            return False
    return True