# -*- coding: utf-8 -*-

import numpy as np

np.random.seed(10)
from sklearn.datasets import load_breast_cancer
from tpot import TPOTClassifier




def get_data():
    data = load_breast_cancer()
    X_t = data.data
    y_t = data.target
    return X_t, y_t


def get_genes():
    X_c, _ = get_data()
    return X_c


X, y = get_data()

classifier = TPOTClassifier(verbosity=2, population_size=1)
classifier.fit(X, y)

# creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
# toolbox = base.Toolbox()
# toolbox.register("attr_bool", get_genes)
# toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_bool)
# toolbox.register("population", tools.initRepeat, list, toolbox.individual)
#
# POP_SIZE = 10


# def evalOneMax(individual):
#     #  implement classifier
#     return np.mean(individual),
#
#
# toolbox.register("evaluate", evalOneMax)
# toolbox.register("mate", np.add)
# toolbox.register("mutate", np.multiply, 1)
# toolbox.register("select", tools.selTournament, tournsize=3)
#
# random.seed(100)
# pop = toolbox.population(n=POP_SIZE)
# CXPB, MUTPB = 0.5, 0.2
#
# print("Start of evolution")
# fitnesses = list(map(toolbox.evaluate, pop))
# for ind, fit in zip(pop, fitnesses):
#     ind.fitness.values = fit
# print("Evaluated %i individuals" % len(pop))
#
# fits = [ind.fitness.values[0] for ind in pop]
#
# g = 0
# # Begin the evolution
# while max(fits) < 100 and g < 10000:
#     # A new generation
#     g = g + 1
#     print("-- Generation %i --" % g)
#
#     # Select the next generation individuals
#     offspring = toolbox.select(pop, len(pop))
#     # Clone the selected individuals
#     offspring = list(map(toolbox.clone, offspring))
#
#     # Apply crossover and mutation on the offspring
#     for child1, child2 in zip(offspring[::2], offspring[1::2]):
#
#         # cross two individuals with probability CXPB
#         if random.random() < CXPB:
#             toolbox.mate(child1, child2)
#
#             # fitness values of the children
#             # must be recalculated later
#             del child1.fitness.values
#             del child2.fitness.values
#
#     for mutant in offspring:
#
#         # mutate an individual with probability MUTPB
#         if random.random() < MUTPB:
#             toolbox.mutate(mutant)
#             del mutant.fitness.values
#
#     # Evaluate the individuals with an invalid fitness
#     invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
#     fitnesses = map(toolbox.evaluate, invalid_ind)
#     for ind, fit in zip(invalid_ind, fitnesses):
#         ind.fitness.values = fit
#
#     print("  Evaluated %i individuals" % len(invalid_ind))
#
#     # The population is entirely replaced by the offspring
#     pop[:] = offspring
#
#     # Gather all the fitnesses in one list and print the stats
#     fits = [ind.fitness.values[0] for ind in pop]
#
# print("-- End of (successful) evolution --")
#
# best_ind = tools.selBest(pop, 1)[0]
# # print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
