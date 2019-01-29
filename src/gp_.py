#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"

import random
import numpy as np
import networkx as nx

from utils_ import reshape_data
from constants import *



# crossover / mate function for two individuals
def mate(individual_1, individual_2):
    # Retain features with importance more than the average relevance and merge two individuals.
    # Future Work : come up with 'intelligent' mating technique. Mating between individuals should be based on the 'attraction'
    # i.e Good 'looking' individuals should mate with other good 'looking' (accuracy) individuals
    offspring_1 = squeeze_individual(individual_1)
    offspring_2 = squeeze_individual(individual_2)
    merge(offspring_1, offspring_2)
    invalidate_fitness(offspring_1)
    filtering(offspring_1)
    return offspring_1





# After merging, fitness and feature_importance have been invalidated.
# They will be recomputed later in BaseFeatureEngineer.evolve method
def invalidate_fitness(individual):
    individual.fitness = 0
    for k in range(len(individual.meta_data)):
        individual.meta_data[k][F_IMP] = -1





# Future Work: Find correlation and remove highly correlated features
# IMPORTANT : Care has to be taken about updating indices of metadata accordingly when a particular feature is removed
def filtering(individual, corr_threshold=0.70):
    pass
    # indices = []
    # for i in range(len(individual.meta_data)):
    #     for j in range(i+1, len(individual.meta_data)):
    #         if set(individual.meta_data[i][A_GRAPH]) == set(individual.meta_data[j][A_GRAPH]):
    #             indices.append(j)
    # for index in indices:
    #     del individual.meta_data[index]
    #     individual.data = np.delete(individual.data, index, axis=1)





# Extract subset of features which have importance greater than the threshold
# Can be used independently as a feature extractor
def squeeze_individual(individual):
    indices = []
    meta_data = {}

    # find average feature importance with initial seed zero
    avg_imp = [0]
    for key in individual.meta_data:
        avg_imp.append(individual.meta_data[key][F_IMP])
    avg_imp = np.mean(avg_imp)

    # remove features with importance less than average importance
    i = 0
    for key in individual.meta_data:
        if individual.meta_data[key][F_IMP] >= avg_imp:
            indices.append(key)
            meta_data[i] = individual.meta_data[key]
            i = i + 1
    data = individual.data[:, indices]

    # if there's a mismatch between data and meta_data
    if data.shape[1] != len(meta_data):
        raise Exception("Mismatch found between data and its meta information.!")
    return Individual(data, meta_data)





# Useful tool for merging two offsprings during mating
# Combines two individuals
def merge(individual_1, individual_2):
    insert_index = individual_1.data.shape[1]
    for i in range(len(individual_2.meta_data)):
        individual_1.meta_data[insert_index] = individual_2.meta_data[i]
        insert_index = insert_index + 1
    individual_1.data = np.append(individual_1.data, individual_2.data, axis=1)
    if individual_1.data.shape[1] != len(individual_1.meta_data):
        raise Exception("Mismatch found between data and its meta information.!")





# Future Work: Reinforcement Learning
# Mutate/ Transform an individual with unary, binary / higher order transforms
def mutate(transformers, individual):
    key = random.choice(list(transformers.keys()))
    # Future Work : Decorators for pre and post sanity check
    # Example : Features are not compatible, Go out of bound after squaring many times etc.
    if transformers[key].param_count == 1:
        # Future Work: Do not apply UnaryTransformation on all features. select features 'intelligently'
        for index in range(len(individual.meta_data)):
            transformers[key].transform(individual=individual, index=index, feat_imp=-1)
    return individual






# select best individuals after cross over and mutation
# Future Work : 'Intelligently' make selection. / create exponential decay in selection
# Now, Top 90% and random 5% (lucky few)
def select(pop, top=0.80, lucky=0.05):
    pop.sort(key=lambda x: x.fitness, reverse=True)
    top_index = int(len(pop) * top)
    top_individuals = pop[0:top_index]
    lucky_indices = np.random.randint(top_index, len(pop), int(len(pop) * lucky))
    lucky_individuals =  [pop[i] for i in lucky_indices]
    top_individuals.extend(lucky_individuals)
    return top_individuals



# combining individual feature graphs into one final graph
# This is used at the end of evolution to show how top_individual evolved
def compose_graphs(individual):
    if individual.meta_data:
        F = individual.meta_data[0][A_GRAPH]
        for i in range(1, len(individual.meta_data)):
            G = individual.meta_data[i][A_GRAPH]
            F = nx.compose(F, G)
        individual.transformation_graph = F





# meta_data should be of the form :
# meta_data = { 'column_index' : {'node_name' : str() instance,
#                                'feature_importance': float() instance between 0 and 1,
#                                 'ancestor_graph': networkx.DiGraph() instance,
#                                },
#             }
def create_metadata(indices, node_names, feature_importance, ancestor_graphs):
    # All four inputs should be iterables.
    metadata = {}
    for index, node_name, feat_imp, ancestor_graph in zip(indices, node_names, feature_importance, ancestor_graphs):
        metadata[index] = {N_NAME: node_name, F_IMP: feat_imp, A_GRAPH: ancestor_graph}
    return metadata




class Individual:
    __slots__ = ['data', 'fitness', 'meta_data', 'transformation_graph']

    def __init__(self, data, meta_data, fitness=0):
        self.data = data
        self.fitness = fitness
        self.meta_data = meta_data
        # final graph composed of all individual feature graphs
        self.transformation_graph = None
        reshape_data(self)



