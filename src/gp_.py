#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"

import random

import numpy as np


# crossover / mate function for two individuals
def mate(individual_1, individual_2, relevance=0.25):
    # Retain features with importance more than the relevance and merge two individuals.
    # Future Work : come up with 'intelligent' mating technique. Mating between individuals should be based on the 'attraction'
    # i.e Good 'looking' individuals should mate with other good 'looking' (accuracy) individuals
    offspring_1 = squeeze_individual(individual_1, relevance)
    offspring_2 = squeeze_individual(individual_2, relevance)
    offspring_1.merge(offspring_2)
    return offspring_1


# Extract subset of features which have importance greater than the threshold
def squeeze_individual(individual, relevance):
    indices = []
    meta_data = {}
    i = 0
    for key in individual.meta_data:
        if individual.meta_data[key]['feature_importance'] > relevance:
            indices.append(key)
            meta_data[i] = individual.meta_data[key]
            i = i + 1
    data = individual.data[:, indices]
    if data.shape[1] != len(meta_data):
        raise Exception("Mismatch found between data and its meta information.!")
    return Individual(data, meta_data)



# Future Work: Reinforcement Learning
def mutate(transformers, individual):
    key = random.choice(list(transformers.keys()))
    # Future Work : Decorators for pre and post sanity check
    # Example : Features are not compatible, Go out of bound after squaring many times etc.
    if transformers[key].param_count == 1:
        # Future Work: Do not apply UnaryTransformation on all features. select features 'intelligently'
        node_names = []
        indices = []
        for index in range(len(individual.meta_data)):
            indices.append(index)
            node_names.append(individual.meta_data[index]['node_name'])
        transformers[key].transform(individual, indices, node_names)
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
        metadata[index] = {'node_name': node_name, 'feature_importance': feat_imp, 'ancestor_graph': ancestor_graph}
    return metadata


class Individual:
    __slots__ = ['data', 'fitness', 'meta_data']

    def __init__(self, data, meta_data, fitness=-1):
        self.data = data
        if self.data.ndim == 1:
            self.data = np.reshape(self.data, (self.data.shape[0], 1))
        self.fitness = fitness
        self.meta_data = meta_data

    def extract_features(self):
        pass

    # Useful tool for merging two offsprings during mating
    def merge(self, individual):
        insert_index = self.data.shape[1]
        for i in range(len(individual.meta_data)):
            self.meta_data[insert_index] = individual.meta_data[i]
        self.data = np.append(self.data, individual.data, axis=1)
        if self.data.shape[1] != len(self.meta_data):
            raise Exception("Mismatch found between data and its meta information.!")
