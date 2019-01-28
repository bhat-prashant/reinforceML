#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"

# Future Work:
# Do NOT apply same transformation again and again !!
# check number of rows in a column. if they are not equal to original number of rows, discard
import numpy as np
from constants import *

def unary_decorator(func):
    def pre_post_check(*args, **kwargs):
        index = kwargs.get(INX)
        transformer = args[0]
        individual = kwargs.get(IND)
        check_pass = False

        # Pre-check
        if transformer.name == SQRT:
            if not np.any(individual.data[:, index] < 0):
                nodes = list(individual.meta_data[index][A_GRAPH])
                if not [node for node in nodes if SQRT in node]:
                    check_pass = True


        if check_pass:
            func(*args, **kwargs)
            # Post-Check
    return pre_post_check


