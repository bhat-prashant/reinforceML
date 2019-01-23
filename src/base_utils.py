import logging
import numpy as np
import random
from data_utils import validate_inputs
from deap import creator
from deap import base
from transformer import get_transformers
from transformer import UnaryTransformer, BinaryTransformer, HigherOrderTransformer

class BaseFeatureEngineer():

    def __init__(self, generation=5, pop_size=None, mutation_rate=0.8,
                 crossover_rate=0.2, scoring='accuracy', upsample=False,
                 downsample=False, random_state=None, verbosity=0, subsample=True):
        self.generation = generation
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.scoring = scoring
        self.upsample = upsample
        self.downsample = downsample
        self.subsample = subsample
        self.random_state = random_state
        self.verbosity = verbosity
        self.feature_count = None
        self.chromosome_count = None
        self.pop = None
        self.transformers = get_transformers()

    def set_random_state(self):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            random.seed(self.random_state)
        else:
            np.random.seed(10)
            random.seed(10)

    def fit(self, X, y):
        self.X = X
        self.y = y
        if validate_inputs(self.X, self.y):
            self.feature_count = X.shape[1]
            self.setup_toolbox()
            if self.pop_size is None:
                self.pop_size = self.feature_count
            elif self.pop_size < self.feature_count:
                raise Exception(" Population size cannot be less than the number of features in the dataset")
            self.pop = self.toolbox.population()
            pass

    def create_population(self):
        population = []
        for column in range(self.X.shape[1]):
            population.append(self.create_individual(self.X[:, column]))
        remaining = self.pop_size - self.feature_count
        i = 0
        while(remaining > 0):
            # TODO Intelligently select transformations in the beginning based on feature's meta information like datetime etc
            key = random.choice(list(self.transformers.keys()))
            if isinstance(self.transformers[key], UnaryTransformer):
                ind_column = self.transformers[key].transform(population[i].value)
                population.append(self.create_individual(ind_column))
                # TODO - Binary and higher order transform
            elif isinstance(self.transformers[key], BinaryTransformer):
                pass
            elif isinstance(self.transformers[key], HigherOrderTransformer):
                pass
            else:
                raise Exception("Unknown transformer type : ", type(self.transformers[key].__name__))
            remaining = remaining - 1
            i = i + 1
        return population



    def create_individual(self, column):
        feature = Feature(column)
        return feature

    def setup_toolbox(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0))
        creator.create("Individual", Feature, fitness=creator.FitnessMax)
        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", self.create_individual, creator.Individual)
        self.toolbox.register("population", self.create_population)



class Feature():
    def __init__(self, column):
        self.transformer_list = []
        self.compute_meta_features(column)
        self.fitness = 0

    def compute_meta_features(self, column):
        if isinstance(column, np.ndarray):
            self.value = column
            self.mean = np.mean(column)
            self.variance = np.var(column)
             # TODO add other meta features about each feature / chromosome






































