#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"

import numpy as np


# Future Work : Write decorator for checking whether generated individual is valid
# Future Work: Reinforcement learning
def grow_individual(pset, min_=1, max_=5):
    height = np.random.randint(min_, max_)
    individual = []
