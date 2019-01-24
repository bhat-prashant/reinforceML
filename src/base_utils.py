import logging
import numpy as np
import multiprocessing
import random
from data_utils import validate_inputs
from deap import creator
from deap import base
from transformer import get_transformers
from transformer import UnaryTransformer, BinaryTransformer, HigherOrderTransformer
from metrics import evaluate, fitness_score, mate, mutate
import copy


class BaseFeatureEngineer():
    __slots__ = ['generation', 'pop_size', 'mutation_rate', 'crossover_rate', 'scoring',
                 'upsample', 'downsample', 'subsample', 'random_state', 'verbosity',
                 'feature_count', 'chromosome_count', 'pop', 'transformers', 'pool',
                 'toolbox', 'initial_score', 'X', 'y']

    def __init__(self, generation=5, pop_size=None, mutation_rate=0.3,
                 crossover_rate=0.7, scoring='accuracy', upsample=False,
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
        self.initial_score = None
        self.chromosome_count = None
        self.pop = None
        self.transformers = get_transformers()
        self.pool = multiprocessing.Pool()
        self.toolbox = None
        self.X = None
        self.y = None

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
            individual = self.create_individual(self.X[:, column])
            individual.fitness, individual.feature_importance = evaluate(individual, self.y)
            population.append(individual)
        remaining = self.pop_size - self.feature_count
        i = 0
        while remaining > 0:
            # Future Work: Intelligently select transformations in the beginning based on feature's meta information like datetime etc
            # Future Work: Initially, features with lesser 'importance' can be transformed first
            key = random.choice(list(self.transformers.keys()))
            trans = self.transformers[key]
            if isinstance(trans, UnaryTransformer):
                data_transformed = trans.transform(population[i].data)
                new_individual = copy.deepcopy(population[i])
                new_individual.data = data_transformed
                new_individual.fitness, new_individual.feature_importance = evaluate(new_individual, self.y)
                for j in range(len(new_individual.feature_importance)):
                    new_individual.features[j].transformer_list.append(trans.name)
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
    def create_individual(data):
        ind = Individual(data)
        return ind

    def setup_toolbox(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0))
        creator.create("Individual", Individual, fitness=creator.FitnessMax)
        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", self.create_individual, creator.Individual)
        self.toolbox.register("population", self.create_population)
        self.toolbox.register("evaluate", evaluate)
        self.toolbox.register("mate", mate)
        self.toolbox.register("map", self.pool.map)
        self.toolbox.register("mutate", mutate, self.transformers)
        # self.toolbox.register("select", tools.selTournament, tournsize=3)

    def fit_init(self):
        self.feature_count = self.X.shape[1]
        self.initial_score, _ = fitness_score(self.X, self.y)
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





class Individual:

    __slots__ = ['fitness', 'data', 'feature_importance', 'features']

    def __init__(self, data):
        self.fitness = 0
        self.data = data
        self.feature_importance = []
        self.features = []
        self.extract_features()

    def extract_features(self):
        if self.data.ndim == 1:
            self.data = np.reshape(self.data, (self.data.shape[0], 1))
        for i in range(self.data.shape[1]):
            feat = Feature(i)
            self.features.append(feat)
            self.feature_importance.append(0)




class Feature:

    __slots__ = ['transformer_list', 'index']

    def __init__(self, index):
        self.transformer_list = []
        self.index = index































