#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"

import random
import numpy as np
import networkx as nx

from utils_ import reshape_data
from constants import *

# crossover / mate function for two individuals
def mate(individual_1, individual_2, relevance=0.25):
    # Retain features with importance more than the relevance and merge two individuals.
    # Future Work : come up with 'intelligent' mating technique. Mating between individuals should be based on the 'attraction'
    # i.e Good 'looking' individuals should mate with other good 'looking' (accuracy) individuals
    offspring_1 = squeeze_individual(individual_1, relevance)
    offspring_2 = squeeze_individual(individual_2, relevance)
    merge(offspring_1, offspring_2)
    invalidate_fitness(offspring_1)
    feature_selection(offspring_1)
    return offspring_1



# After merging, fitness and feature_importance have been invalidated.
# They will be recomputed later in BaseFeatureEngineer.evolve method
def invalidate_fitness(individual):
    individual.fitness = 0
    for k in range(len(individual.meta_data)):
        individual.meta_data[k][F_IMP] = -1



# Find correlation and remove highly correlated features
def feature_selection(individual, corr_threshold=0.70):
    corr_coef = np.corrcoef(individual.data, rowvar=False)
    if np.any(corr_coef > corr_threshold):
        pass
    pass



# Extract subset of features which have importance greater than the threshold
def squeeze_individual(individual, relevance):
    indices = []
    meta_data = {}
    i = 0
    for key in individual.meta_data:
        if individual.meta_data[key][F_IMP] > relevance:
            indices.append(key)
            meta_data[i] = individual.meta_data[key]
            i = i + 1
    data = individual.data[:, indices]
    if data.shape[1] != len(meta_data):
        raise Exception("Mismatch found between data and its meta information.!")
    return Individual(data, meta_data)



# Useful tool for merging two offsprings during mating
def merge(individual_1, individual_2):
    insert_index = individual_1.data.shape[1]
    for i in range(len(individual_2.meta_data)):
        individual_1.meta_data[insert_index] = individual_2.meta_data[i]
        insert_index = insert_index + 1
    individual_1.data = np.append(individual_1.data, individual_2.data, axis=1)
    if individual_1.data.shape[1] != len(individual_1.meta_data):
        raise Exception("Mismatch found between data and its meta information.!")



# Future Work: Reinforcement Learning
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
        self.transformation_graph = None
        reshape_data(self)



