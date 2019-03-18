#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"

import numpy as np
from transformer import ScaledArray, SelectedArray, ExtractedArray, ClassifiedArray
from copy import deepcopy
from collections import defaultdict
from deap import tools, algorithms
from tqdm import tqdm


# Future Work : Write decorator for checking whether generated individual is valid
# Future Work: Reinforcement learning
def grow_individual(pset, trans_types, min_=3, max_=8):
    height = np.random.randint(low=len(trans_types), high=max_)
    individual = []
    idx = 0
    # Add unary operators
    if 'unary' in trans_types:
        # first 'unary' with input_matrix
        transformer = np.random.choice(pset.primitives[np.ndarray])
        individual.append(transformer)
        for arg_type in transformer.args[idx:]:
            terminal = np.random.choice(pset.terminals[arg_type])
            individual.append(terminal)
        height = height - 1
        idx = 1

        # subsequent unary operators without input_matrix
        while height >= len(trans_types):
            prim = []
            transformer = np.random.choice(pset.primitives[np.ndarray])
            individual = [transformer] + individual
            # input_matrix is skipped since it was included before
            for arg_type in transformer.args[1:]:
                terminal = np.random.choice(pset.terminals[arg_type])
                prim.append(terminal)
            # append to the primitive list
            individual = individual + prim
            height = height - 1

    if 'scaler' in trans_types:
        scaler = np.random.choice(pset.primitives[ScaledArray])
        individual = [scaler] + individual
        for arg_type in scaler.args[idx:]:
            terminal = np.random.choice(pset.terminals[arg_type])
            individual.append(terminal)
        idx = 1
    if 'selector' in trans_types:
        selector = np.random.choice(pset.primitives[SelectedArray])
        individual = [selector] + individual
        for arg_type in selector.args[idx:]:
            terminal = np.random.choice(pset.terminals[arg_type])
            individual.append(terminal)
        idx = 1
    if 'extractor' in trans_types:
        extractor = np.random.choice(pset.primitives[ExtractedArray])
        individual = [extractor] + individual
        for arg_type in extractor.args[idx:]:
            terminal = np.random.choice(pset.terminals[arg_type])
            individual.append(terminal)
        idx = 1
    if 'classifier' in trans_types:
        classifier = np.random.choice(pset.primitives[ClassifiedArray])
        individual = [classifier] + individual
        for arg_type in classifier.args[idx:]:
            terminal = np.random.choice(pset.terminals[arg_type])
            individual.append(terminal)
        idx = 1
    # individual as a list (iterable)
    return individual


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


 # Borrowed directly from DEAP, credits: DEAP
# There are two reasons 1. tqdm can be used and 2. surrogate RL model can be trained further
# Edited parts : Aafter each generation eval(), train RL model. That's all!
def eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                       stats=None, halloffame=None, verbose=__debug__):

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # -----------------
    toolbox.train_RL()
    # -----------------

    # Begin the generational process
    for gen in tqdm(range(1, ngen + 1)):
        # Vary the population
        offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # -----------------
        toolbox.train_RL()
        # -----------------

    return population, logbook




















