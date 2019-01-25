import copy
import multiprocessing
import random
import numpy as np
from deap import creator, base, algorithms, tools
from sklearn.metrics import accuracy_score

from data_util import validate_inputs
from gp_util import mate, mutate, select, Individual
from transformer import get_transformers, UnaryTransformer, BinaryTransformer, HigherOrderTransformer
from metrics import evaluate, fitness_score



class BaseFeatureEngineer:

    __slots__ = ['generation', 'pop_size', 'mutation_rate', 'crossover_rate', 'scorer',
                 'upsample', 'downsample', 'subsample', 'random_state', 'verbosity',
                 'feature_count', 'chromosome_count', 'pop', 'transformers', 'pool',
                 'toolbox', 'initial_score', 'X', 'y']

    def __init__(self, generation=5, pop_size=None, mutation_rate=0.3,
                 crossover_rate=0.7, scorer=accuracy_score, upsample=False,
                 downsample=False, random_state=None, verbosity=0, subsample=True):
        self.generation = generation
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.scorer = scorer
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
            self.random_state = 10
            np.random.seed(10)
            random.seed(10)


    def create_population(self):
        population = []
        for column in range(self.X.shape[1]):
            individual = self.create_individual(self.X[:, column])
            individual.fitness, individual.feature_importance = evaluate(individual, self.y, scorer=self.scorer)
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
                new_individual.fitness, new_individual.feature_importance = evaluate(new_individual, self.y,
                                                                                     scorer=self.scorer)
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
        creator.create("FitnessMax", base.Fitness, weights=1.0)
        creator.create("Individual", Individual, fitness=creator.FitnessMax)
        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", self.create_individual, creator.Individual)
        self.toolbox.register("population", self.create_population)
        self.toolbox.register("evaluate", evaluate)
        self.toolbox.register("mate", mate)
        self.toolbox.register("map", self.pool.map)
        self.toolbox.register("mutate", mutate, self.transformers)
        self.toolbox.register("select", select)

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


    def evolve(self):
        # Begin the generational process
        for gen in range(1, self.generation + 1):
            # Select the next generation individuals
            offspring = self.toolbox.select(self.pop, top=0.90)
            for i in range(1, len(offspring), 2):
                if random.random() < self.crossover_rate:
                    offspring[i] = self.toolbox.mate(offspring[i - 1], offspring[i])
                # Slightly different from original algorithm
                if random.random() < self.mutation_rate:
                    offspring[i] = self.toolbox.mutate(offspring[i])

            for ind in offspring:
                if not ind.features:
                    offspring.remove(ind)
                    continue
                if ind.fitness == 0:
                    ind.fitness, ind.feature_importance = self.toolbox.evaluate(ind, self.y, scorer=self.scorer)

            self.pop[:] = offspring
        return sorted(self.pop, key=lambda x: x.fitness, reverse=True)[0]

    def fit(self, X, y):
        self.set_random_state()
        if validate_inputs(X, y):
            self.X = X
            self.y = y
            self.fit_init()
            best_ind = self.evolve()
            print("Initial score : ", self.initial_score)
            print("Best Fitness : ", best_ind.fitness)





































