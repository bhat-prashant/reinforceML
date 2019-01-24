import logging
import numpy as np
import random
from data_utils import validate_inputs
from deap import creator
from deap import base
from transformer import get_transformers
from transformer import UnaryTransformer, BinaryTransformer, HigherOrderTransformer
from metrics import evaluate, fitness_score
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


    def create_population(self):
        population = []
        for column in range(self.X.shape[1]):
            population.append(self.create_individual(self.X[:, column]))
        remaining = self.pop_size - self.feature_count
        i = 0
        while remaining > 0:
            # Future Work: Intelligently select transformations in the beginning based on feature's meta information like datetime etc
            key = random.choice(list(self.transformers.keys()))
            trans = self.transformers[key]
            if isinstance(trans, UnaryTransformer):
                ind_column = trans.transform(population[i].value)
                individual = self.create_individual(ind_column)
                individual.transformer_list.append(trans)
                population.append(individual)
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
    def create_individual(column):
        feature = Feature(column)
        return feature

    def setup_toolbox(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0))
        creator.create("Individual", Feature, fitness=creator.FitnessMax)
        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", self.create_individual, creator.Individual)
        self.toolbox.register("population", self.create_population)
        self.toolbox.register("evaluate", evaluate)
        # self.toolbox.register("mate", tools.cxTwoPoint)
        # self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        # self.toolbox.register("select", tools.selTournament, tournsize=3)

    def fit_init(self):
        self.feature_count = self.X.shape[1]
        self.initial_score = fitness_score(self.X, self.y)
        self.setup_toolbox()
        if self.pop_size is None:
            self.pop_size = self.feature_count
        elif self.pop_size < self.feature_count:
            raise Exception(" Population size cannot be less than the number of features in the dataset")
        self.pop = self.toolbox.population()
        pass


    def fit(self, X, y):
        self.set_random_state()
        if validate_inputs(X, y):
            self.X = X
            self.y = y
            self.fit_init()




class Feature():

    def __init__(self, column):
        self.transformer_list = []
        self.compute_meta_features(column)
        self.fitness = 0

    # For now, it is assumed that input features are numerical !!
    # Future Work: add other meta information about each feature / chromosome
    def compute_meta_features(self, column):
        if isinstance(column, np.ndarray):
            self.value = column
            self.mean = np.mean(column)
            self.variance = np.var(column)
        else:
            raise Exception("Unknown data format. Expected \'numpy.ndarray\', Instead got {}", type(column))







































