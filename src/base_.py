#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"

import copy
import multiprocessing
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from deap import creator, base
from sklearn.metrics import accuracy_score

from data_ import validate_inputs
from gp_ import mate, mutate, select, create_metadata, compose_graphs, squeeze_individual, Individual
from metrics_ import evaluate, fitness_score
from transformer_ import get_unary_transformers, get_universal_transformers, \
    UnaryTransformer, BinaryTransformer, HigherOrderTransformer
from constants import *

class BaseFeatureEngineer:
    __slots__ = ['_generation', '_pop_size', '_mutation_rate', '_crossover_rate', '_scorer',
                 '_upsample', '_downsample', '_subsample', '_random_state', '_verbosity',
                 '_feature_count', '_chromosome_count', '_pop', '_unary_transformers', '_universal_transformers',
                 '_pool', '_toolbox', '_initial_score', '_X', '_y', 'append_original']



    def __init__(self, generation=5, pop_size=None, mutation_rate=0.3,
                 crossover_rate=0.7, scorer=accuracy_score, upsample=False,
                 downsample=False, random_state=None, verbosity=0, subsample=True, append_original=True):
        self._generation = generation
        self._pop_size = pop_size
        self._mutation_rate = mutation_rate
        self._crossover_rate = crossover_rate
        self._scorer = scorer
        self._upsample = upsample
        self._downsample = downsample
        self._subsample = subsample
        self._random_state = random_state
        self._verbosity = verbosity
        self._feature_count = None
        self._initial_score = None
        self._chromosome_count = None
        self._pop = None
        self._unary_transformers = get_unary_transformers()
        self._universal_transformers = get_universal_transformers()
        self._pool = multiprocessing.Pool()
        self._toolbox = None
        self._X = None
        self._y = None
        self.append_original = append_original



    def _set_random_state(self):
        if self._random_state is not None:
            np.random.seed(self._random_state)
            random.seed(self._random_state)
        else:
            self._random_state = 10
            np.random.seed(10)
            random.seed(10)



    def _update_fitness(self, individual):
        individual.fitness, feature_importance = \
            evaluate(individual, self._y, scorer=self._scorer)
        for k in range(len(feature_importance)):
            individual.meta_data[k][F_IMP] = feature_importance[k]


    # method to create individuals in bulk
    def _create_population(self):
        population = []

        # Individuals created from original features
        for column in range(self._X.shape[1]):
            fitness, feat_importance = evaluate(self._X[:, column], self._y, scorer=self._scorer)
            G = nx.DiGraph()
            G.add_node(str(column))
            meta_data = create_metadata([0], [str(column)], feat_importance, [G])
            individual = self.create_individual(self._X[:, column], meta_data)
            individual.fitness = fitness
            population.append(individual)

        remaining = self._pop_size - self._feature_count
        i = 0

        # Individuals can be created by transformation. For now, transformations are disabled in the initial population
        while remaining > 0:
            # Future Work: Intelligently select transformations in the beginning based
            # on feature's meta information like datetime etc
            # Future Work: Initially, features with lesser 'importance' can be transformed first
            key = random.choice(list(self._unary_transformers.keys()))
            trans = self._unary_transformers[key]

            if isinstance(trans, UnaryTransformer):
                # Each individual has only one feature. Therefore, feat_imp is set to 1
                new_individual = copy.deepcopy(population[i])
                # trans.transform(individual=new_individual, index=0, feat_imp=1)
                # self._update_fitness(new_individual)
                population.append(new_individual)

                # Future Work: - Binary and higher order transform
            elif isinstance(trans, BinaryTransformer):
                pass
            elif isinstance(trans, HigherOrderTransformer):
                pass
            else:
                raise Exception("Unknown transformer type : ", type(trans.__name__))
            remaining = remaining - 1
            i = i + 1

        return population


    # method to create an individual
    @staticmethod
    def create_individual(data, meta_data):
        ind = Individual(data, meta_data)
        return ind


    # Evolution toolbox
    def _setup_toolbox(self):
        creator.create("FitnessMax", base.Fitness, weights=1.0)
        creator.create("Individual", Individual, fitness=creator.FitnessMax)
        self._toolbox = base.Toolbox()
        self._toolbox.register("individual", self.create_individual, creator.Individual)
        self._toolbox.register("population", self._create_population)
        self._toolbox.register("evaluate", evaluate)
        self._toolbox.register("mate", mate)
        self._toolbox.register("map", self._pool.map)
        self._toolbox.register("mutate", mutate, self._unary_transformers)
        self._toolbox.register("select", select)


    # Initialization steps
    def _fit_init(self):
        self._feature_count = self._X.shape[1]

        # initial accuracy on the given dataset
        self._initial_score, _ = fitness_score(self._X, self._y)

        # setup toolbox for evolution
        self._setup_toolbox()

        # Make sure population is at least as big as original features
        if self._pop_size is None:
            self._pop_size = self._feature_count
        elif self._pop_size < self._feature_count:
            raise Exception("Population size should not be less than the number of features in the data set")

        # create population
        self._pop = self._toolbox.population()




    def _evolve(self):
        # Begin the generational process
        for gen in range(1, self._generation + 1):
            # Select the next generation individuals
            offspring = self._toolbox.select(self._pop, top=0.80)

            for i in range(1, len(offspring), 2):
                # Mate / cross-over
                if random.random() < self._crossover_rate:
                    offspring[i] = self._toolbox.mate(offspring[i - 1], offspring[i])
                # Slightly different from original algorithm
                # mutate
                if random.random() < self._mutation_rate:
                    offspring[i] = self._toolbox.mutate(offspring[i])

            for ind in offspring:
                # remove individual if feature set is empty
                if not ind.meta_data:
                    offspring.remove(ind)
                    continue
                # update accuracy / fitness score after mating, mutation
                if ind.fitness <= 0:
                    self._update_fitness(ind)

            # update population
            self._pop[:] = offspring

        # return top 3 individuals
        return sorted(self._pop, key=lambda x: x.fitness, reverse=True)[:3]



    def fit(self, X, y):
        self._set_random_state()
        # Future Work: checking OneHotEncoding, datetime etc
        if validate_inputs(X, y):
            self._X = X
            self._y = y
            self._fit_init()



    def _append_to_original(self, top_individual):
        # Future Work: When original features are appended with transformed ones, update transformation_graph accordingly
        print("Initial score : ", self._initial_score)

        # append original data
        top_individual.data = np.append(top_individual.data, self._X, axis=1)
        top_individual.fitness, fea_imp = fitness_score(top_individual.data, self._y)

        # add meta_data and update feature importance
        meta_length = len(top_individual.meta_data)
        for j in range(0, self._X.shape[1]):
            G = nx.DiGraph()
            G.add_node(str(j))
            top_individual.meta_data[meta_length + j] = {N_NAME: str(j), F_IMP: fea_imp[j], A_GRAPH: G }
        for k in range(meta_length):
            top_individual.meta_data[k][F_IMP] = fea_imp[k]

        # Feature selection - Select top features with relevance greater than the average relevance
        top_individual = squeeze_individual(top_individual)
        top_individual.fitness, _ = fitness_score(top_individual.data, self._y)

        # check if universal transformation yields better results
        fitness, trans_name = 0, None
        for key in self._universal_transformers:
            data = self._universal_transformers[key].transform(individual=top_individual)
            fitness, _ = fitness_score(data, self._y)
            if fitness > top_individual.fitness:
                trans_name = key

        # apply universal transformation
        if not trans_name is None:
            data = self._universal_transformers[trans_name].transform(individual=top_individual)
            top_individual.fitness = fitness
            for index in range(len(top_individual.meta_data)):
                node_name = top_individual.meta_data[index][N_NAME]
                top_individual.meta_data[index][A_GRAPH].add_node(trans_name)
                top_individual.meta_data[index][A_GRAPH].add_edge(node_name, trans_name, transformer=trans_name)

        # display results
        self._compose_display(top_individual)
        print("Best Fitness : ", top_individual.fitness, '\n')



    # Combine individual feature graphs and display it as a final transformation_graph
    def _compose_display(self, individual):
        compose_graphs(individual)
        if individual.transformation_graph is not None:
            nx.draw(individual.transformation_graph, with_labels=True)
            plt.show()



    def transform(self):
        top_individuals = self._evolve()
        for top_individual in top_individuals:
            # If you would like to combine original feature set with transformed set
            if self.append_original:
                self._append_to_original(top_individual)
            else:
                self._compose_display(top_individual)
                print("Initial score : ", self._initial_score)
                print("Best Fitness : ", top_individual.fitness, '\n')


