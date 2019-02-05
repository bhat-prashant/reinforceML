#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"

import numpy as np


# Future Work : Write decorator for checking whether generated individual is valid
# Future Work: Reinforcement learning
def grow_individual(pset, min_=1, max_=5):
    height = np.random.randint(low=min_, high=max_)
    primitive = []

    # first primitive with input_matrix
    transformer = np.random.choice(pset.primitives[np.ndarray])
    primitive.append(transformer)
    for arg_type in transformer.args[0:]:
        terminal = np.random.choice(pset.terminals[arg_type])
        primitive.append(terminal)

    # subsequent primitives without input_matrix
    while height > 2:
        prim = []
        transformer = np.random.choice(pset.primitives[np.ndarray])
        primitive = [transformer] + primitive
        # input_matrix is skipped since it was included before
        for arg_type in transformer.args[1:]:
            terminal = np.random.choice(pset.terminals[arg_type])
            prim.append(terminal)
        # append to the primitive list
        primitive = primitive + prim
        height -= 1


    # root with Output_array
    individual = []
    type_ = pset.ret
    universal = np.random.choice(pset.primitives[type_])
    individual.append(universal)
    # append all primitives to root
    individual =  individual + primitive
    # input_matrix is skipped since it was included before
    for arg_type in universal.args[1:]:
        terminal = np.random.choice(pset.terminals[arg_type])
        individual.append(terminal)

    # individual as a list (iterable)
    return individual

