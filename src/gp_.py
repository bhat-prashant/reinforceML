#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"

import numpy as np
from transformer import ScaledArray, SelectedArray, ExtractedArray
from copy import deepcopy
from collections import defaultdict
from deap import tools, algorithms
from tqdm import tqdm

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
        height = height - 1

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
            height = height - 1

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


# borrowed directly from DEAP, credits: DEAP
# There are two reasons 1. tqdm can be used and 2. surrogate RL model can be trained further
def eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                   stats=None, halloffame=None, verbose=__debug__):
    """This is the :math:`(\mu + \lambda)` evolutionary algorithm.

    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param mu: The number of individuals to select for the next generation.
    :param lambda\_: The number of children to produce at each generation.
    :param cxpb: The probability that an offspring is produced by crossover.
    :param mutpb: The probability that an offspring is produced by mutation.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution.

    The algorithm takes in a population and evolves it in place using the
    :func:`varOr` function. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evalutions for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The *cxpb* and *mutpb* arguments are passed to the
    :func:`varOr` function. The pseudocode goes as follow ::

        evaluate(population)
        for g in range(ngen):
            offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)
            evaluate(offspring)
            population = select(population + offspring, mu)

    First, the individuals having an invalid fitness are evaluated. Second,
    the evolutionary loop begins by producing *lambda_* offspring from the
    population, the offspring are generated by the :func:`varOr` function. The
    offspring are then evaluated and the next generation population is
    selected from both the offspring **and** the population. Finally, when
    *ngen* generations are done, the algorithm returns a tuple with the final
    population and a :class:`~deap.tools.Logbook` of the evolution.

    This function expects :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox. This algorithm uses the :func:`varOr`
    variation.
    """
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

    return population, logbook






















