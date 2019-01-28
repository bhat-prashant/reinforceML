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
from gp_ import mate, mutate, select, create_metadata, compose_graphs, Individual
from metrics_ import evaluate, fitness_score
from transformer_ import get_transformers, UnaryTransformer, BinaryTransformer, HigherOrderTransformer
from constants import *

class BaseFeatureEngineer:
    __slots__ = ['_generation', '_pop_size', '_mutation_rate', '_crossover_rate', '_scorer',
                 '_upsample', '_downsample', '_subsample', '_random_state', '_verbosity',
                 '_feature_count', '_chromosome_count', '_pop', '_transformers', '_pool',
                 '_toolbox', '_initial_score', '_X', '_y']

    def __init__(self, generation=5, pop_size=None, mutation_rate=0.3,
                 crossover_rate=0.7, scorer=accuracy_score, upsample=False,
                 downsample=False, random_state=None, verbosity=0, subsample=True):
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
        self._transformers = get_transformers()
        self._pool = multiprocessing.Pool()
        self._toolbox = None
        self._X = None
        self._y = None

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

    def _create_population(self):
        population = []

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

        while remaining > 0:
            # Future Work: Intelligently select transformations in the beginning based
            # on feature's meta information like datetime etc
            # Future Work: Initially, features with lesser 'importance' can be transformed first
            key = random.choice(list(self._transformers.keys()))
            trans = self._transformers[key]

            if isinstance(trans, UnaryTransformer):
                # Each individual has only one feature. Therefore, feat_imp is set to 1
                new_individual = copy.deepcopy(population[i])
                trans.transform(individual=new_individual, index=0, feat_imp=1)
                self._update_fitness(new_individual)
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

    @staticmethod
    def create_individual(data, meta_data):
        ind = Individual(data, meta_data)
        return ind

    def _setup_toolbox(self):
        creator.create("FitnessMax", base.Fitness, weights=1.0)
        creator.create("Individual", Individual, fitness=creator.FitnessMax)
        self._toolbox = base.Toolbox()
        self._toolbox.register("individual", self.create_individual, creator.Individual)
        self._toolbox.register("population", self._create_population)
        self._toolbox.register("evaluate", evaluate)
        self._toolbox.register("mate", mate)
        self._toolbox.register("map", self._pool.map)
        self._toolbox.register("mutate", mutate, self._transformers)
        self._toolbox.register("select", select)

    def _fit_init(self):
        self._feature_count = self._X.shape[1]
        self._initial_score, _ = fitness_score(self._X, self._y)
        self._setup_toolbox()
        if self._pop_size is None:
            self._pop_size = self._feature_count
        elif self._pop_size < self._feature_count:
            raise Exception("Population size should not be less than the number of features in the data set")
        self._pop = self._toolbox.population()
        pass

    def _evolve(self):
        # Begin the generational process
        for gen in range(1, self._generation + 1):
            # Select the next generation individuals
            offspring = self._toolbox.select(self._pop, top=0.90)

            for i in range(1, len(offspring), 2):
                if random.random() < self._crossover_rate:
                    offspring[i] = self._toolbox.mate(offspring[i - 1], offspring[i])
                # Slightly different from original algorithm
                if random.random() < self._mutation_rate:
                    offspring[i] = self._toolbox.mutate(offspring[i])

            for ind in offspring:
                if not ind.meta_data:
                    offspring.remove(ind)
                    continue
                if ind.fitness <= 0:
                    self._update_fitness(ind)

            self._pop[:] = offspring
        return sorted(self._pop, key=lambda x: x.fitness, reverse=True)[0]

    def fit(self, X, y):
        self._set_random_state()
        if validate_inputs(X, y):
            self._X = X
            self._y = y
            self._fit_init()

    def transform(self):
        top_individual = self._evolve()
        # Future Work: Combining all individual feature  graphs in to one image and saving it to file.
        compose_graphs(top_individual)
        if top_individual.transformation_graph is not None:
            nx.draw(top_individual.transformation_graph, with_labels=True)
            plt.show()
        print("Initial score : ", self._initial_score)
        print("Best Fitness : ", top_individual.fitness)


