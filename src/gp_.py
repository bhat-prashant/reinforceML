#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"

import numpy as np
from transformer import ScaledArray, SelectedArray, ExtractedArray
from copy import deepcopy
from collections import defaultdict


# Future Work : Write decorator for checking whether generated individual is valid
# Future Work: Reinforcement learning
def grow_individual(pset, min_=3, max_=8):
    height = np.random.randint(low=min_, high=max_)
    individual = []
    idx = 1
    # Add unary operators if height is greater than 3
    if height > 3:
        # first 'unary' with input_matrix
        transformer = np.random.choice(pset.primitives[np.ndarray])
        individual.append(transformer)
        for arg_type in transformer.args[0:]:
            terminal = np.random.choice(pset.terminals[arg_type])
            individual.append(terminal)
        height -= 1

        # subsequent unary operators without input_matrix
        while height > 3:
            prim = []
            transformer = np.random.choice(pset.primitives[np.ndarray])
            individual = [transformer] + individual
            # input_matrix is skipped since it was included before
            for arg_type in transformer.args[1:]:
                terminal = np.random.choice(pset.terminals[arg_type])
                prim.append(terminal)
            # append to the primitive list
            individual = individual + prim
            height -= 1

    elif height <= 3:
        # np.ndarray is included only if it was not included before
        idx = 0
    # one scaler, one selector and one extractor is added to the tree
    scaler = np.random.choice(pset.primitives[ScaledArray])
    individual = [scaler] + individual
    for arg_type in scaler.args[idx:]:
        terminal = np.random.choice(pset.terminals[arg_type])
        individual.append(terminal)

    selector = np.random.choice(pset.primitives[SelectedArray])
    individual = [selector] + individual
    for arg_type in selector.args[1:]:
        terminal = np.random.choice(pset.terminals[arg_type])
        individual.append(terminal)

    extractor = np.random.choice(pset.primitives[ExtractedArray])
    individual = [extractor] + individual
    for arg_type in extractor.args[1:]:
        terminal = np.random.choice(pset.terminals[arg_type])
        individual.append(terminal)

    # individual as a list (iterable)
    return individual

# mating two individuals during evolution
def mate(pset, ind_1, ind_2):
    off_1 = deepcopy(ind_1)
    off_2 = grow_individual(pset)
    return off_1, off_2

# mutate an individual during evolution by randomly replacing parameters
# Future Work : allow individual to grow, shrink during mutation
def mutate(pset, ind):
    individual = deepcopy(ind)
    # always ind[height] represents input_matrix
    pos = individual.height + 1         # terminal position
    idx = individual.height -1          # primitive position
    while idx >= 0:
        for arg_type in individual[idx].args[1:]:
            # pick a random parameter and replace
            individual[pos] = np.random.choice(pset.terminals[arg_type])
            pos += 1
        idx -= 1
    return individual,




# borrowed from tpot, credits: tpot
def cxOnePoint(ind1, ind2):
    """Randomly select in each individual and exchange each subtree with the
    point as root between each individual.
    :param ind1: First tree participating in the crossover.
    :param ind2: Second tree participating in the crossover.
    :returns: A tuple of two trees.
    """
    # List all available primitive types in each individual
    types1 = defaultdict(list)
    types2 = defaultdict(list)

    for idx, node in enumerate(ind1[1:], 1):
        types1[node.ret].append(idx)
    common_types = []
    for idx, node in enumerate(ind2[1:], 1):
        if node.ret in types1 and node.ret not in types2:
            common_types.append(node.ret)
        types2[node.ret].append(idx)

    if len(common_types) > 0:
        type_ = np.random.choice(common_types)

        index1 = np.random.choice(types1[type_])
        index2 = np.random.choice(types2[type_])

        slice1 = ind1.searchSubtree(index1)
        slice2 = ind2.searchSubtree(index2)
        ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]

    return ind1, ind2

























