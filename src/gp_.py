#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"

import numpy as np


# Future Work : Write decorator for checking whether generated individual is valid
# Future Work: Reinforcement learning
def grow_individual(pset, min_=1, max_=5):
    height = np.random.randint(low=min_, high=max_)
    individual = []
    print(height)
    type_ = pset.ret
    universal = np.random.choice(pset.primitives[type_])
    individual.append(universal)
    for arg_type in universal.args:
        terminal = np.random.choice(pset.terminals[arg_type])
        individual.append(terminal)
    while height > 1:
        transformer = np.random.choice(pset.primitives[np.ndarray])
        individual.append(transformer)
        # np ndarray is skipped since it was included before
        for arg_type in transformer.args[1:]:
            terminal = np.random.choice(pset.terminals[arg_type])
            individual.append(terminal)
        height -= 1
    return individual
