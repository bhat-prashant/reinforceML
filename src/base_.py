#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"

import abc
import random

random.seed(10)
import warnings
import numpy as np
from copy import  deepcopy
from deap import gp, creator, base, tools
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from gp_ import grow_individual, mutate, cxOnePoint, eaMuPlusLambda
from utils_ import reshape_numpy
from lookup import TransformerLookUp
from transformer import TransformerClassGenerator, ScaledArray, SelectedArray, ExtractedArray, ClassifiedArray
from ddqn import Replay_Memory

class BaseReinforceML(BaseEstimator, TransformerMixin, metaclass=abc.ABCMeta):

    def __init__(self, estimator, reinforce_learner, feateng, generation, pop_size, mutation_rate,
                 crossover_rate, scorer, inputArray, outputArray, trans_types, random_state):
        """ Base class for tree based evolution

        :param estimator: an instance of sklearn estimator
        :param reinforce_learner: surrogate model for informed evolution
                It is an instance of skelarn regressor
        :param feateng: boolean,
                if True, task is feature engineering. If False, task is one of regression, classification etc.
        :param generation: int, number of generations to run during evolution
        :param pop_size: int, number of different individuals / pipelines to be created during evolution (population size)
        :param mutation_rate: float, rate of mutation during evolution
        :param crossover_rate: float, rate of cross over during evolution. sum of mutation_rate and crossover_rate should sum to 1
        :param scorer: one of sklearn metrics, usually one of accuracy_score / r2_score depending on the target_type
                can be left to default in case you mention target_type
        :param inputArray: Input array type for primitive set, default to np.ndarray
        :param outputArray: Ouput array type for primitive set
        :param trans_types: list of transformers to be used during evolution
        :param random_seed: int,
                random seed for numpy and random libs
        """
        self._generation = generation
        self._pop_size = pop_size
        self._feateng = feateng
        self._mutation_rate = mutation_rate
        self._crossover_rate = crossover_rate
        self._estimator = estimator
        self._reinforce_learner = reinforce_learner
        self._scorer = scorer
        self._inputArray = inputArray
        self._outputArray = outputArray
        self._trans_types = trans_types
        self._random_state = random_state
        self._feature_count = None
        self._initial_score = None
        self._pop = None
        self._toolbox = None
        self._X = None
        self._y = None
        self._pset = None
        self._columns = None
        self._initialise_replay()


    def _setup_pset(self):
        """ creates a typed PrimitiveSet

        Given transformer types, creates a typed primitive set
        Also creates an empty surrogate dataframe

        :return: None
        """
        if self._inputArray is not None and self._outputArray is not None and isinstance(self._trans_types, list):
            self._pset = gp.PrimitiveSetTyped('MAIN', self._inputArray, self._outputArray)
            self._pset.renameArguments(ARG0='input_matrix')
            lookup = TransformerLookUp(self._feature_count, self._random_state)
            self._columns = []  # input features for RL training
            for type_ in self._trans_types:
                trans_lookup = lookup.get_lookup(type_)
                # add transformers as primitives
                for key in trans_lookup:
                    # add transformers to pset
                    transformer = TransformerClassGenerator(key, trans_lookup[key])
                    if type_ == 'unary':
                        self._pset.addPrimitive(transformer, [np.ndarray] + transformer.arg_types, np.ndarray)
                    elif type_ == 'scaler':
                        self._pset.addPrimitive(transformer, [np.ndarray] + transformer.arg_types, ScaledArray)
                    elif type_ == 'selector':
                        self._pset.addPrimitive(transformer, [np.ndarray] + transformer.arg_types, SelectedArray)
                    elif type_ == 'extractor':
                        self._pset.addPrimitive(transformer, [np.ndarray] + transformer.arg_types, ExtractedArray)
                    elif type_ == 'classifier':
                        self._pset.addPrimitive(transformer, [np.ndarray] + transformer.arg_types, ClassifiedArray)

                    # add transformer arguments as terminal
                    # arg_types is a list
                    for arg in transformer.arg_types:
                        values = list(arg.values)
                        for val in values:
                            arg_name = arg.__name__ + "=" + str(val)
                            self._pset.addTerminal(val, arg, name=arg_name)
                            # input features for RL training
                            self._columns.append(arg_name)

    def _setup_toolbox(self):
        """ sets up toolbox for evolution

        Register essential evolutionary functions with the toolbox

        :return: None
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            creator.create('FitnessMulti', base.Fitness, weights=(1.0,-0.4))
            creator.create('Individual', gp.PrimitiveTree, fitness=creator.FitnessMulti, statistics=dict)
        self._toolbox = base.Toolbox()
        self._toolbox.register('expr', grow_individual, pset=self._pset, trans_types=self._trans_types,
                               random_state=self._random_state, max_=8)
        self._toolbox.register('individual', tools.initIterate, creator.Individual, self._toolbox.expr)
        self._toolbox.register('population', tools.initRepeat, list, self._toolbox.individual)
        self._toolbox.register('evaluate', self._evaluate)
        self._toolbox.register('select', tools.selBest)
        self._toolbox.register('mate', cxOnePoint, self._random_state)
        self._toolbox.register('mutate', mutate, self)

    def _compile_to_sklearn(self, individual):
        """ Transform DEAP Individual to sklearn pipeline

            Note: pipeline is appended with self._estimator as the final step

        :param individual: an instance of DEAP Individual
        :return: sklearn pipeline
        """
        height = individual.height - 1  # start from first primitive
        arg_pos = individual.height + 1  # start from first argument after input_matrix for the above primitive
        pipeline = []
        # for every primitive transformer, do
        while height >= 0:
            # get primitive class
            trans_class = self._pset.context[individual[height].name]
            args = {}

            # get corresponding arguments
            for arg_type in trans_class.arg_types:
                args[arg_type.name] = self._pset.context[individual[arg_pos].name]
                arg_pos = arg_pos + 1

            operator = trans_class.transformer(**args)
            pipeline.append(operator)
            # Start from most basic primitive and move up the tree towards root / universal primitive
            height = height - 1
        # add a estimator to the pipeline only if its a feature engineering
        if self._feateng:
            pipeline.append(self._estimator)
        pipe = make_pipeline(*pipeline)
        return pipe

    def _evolve(self):
        """ Start evolution

        Create Hall Of Fame, register statistics and start the evolution

        :return:None
        """
        print('Start of evolution')
        self._hof = tools.HallOfFame(3)
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        # Future Work: add few new individuals every generation
        pop, log = eaMuPlusLambda(self._pop, toolbox=self._toolbox, mu=self._pop_size, lambda_=self._pop_size,
                                  cxpb=self._crossover_rate, mutpb=self._mutation_rate, ngen=self._generation,
                                  stats=stats, halloffame=self._hof, verbose=True)
        # Future Work: Export Hall of Fame individuals' sklearn-pipeline to a file
        self._solution_to_file()



    def fit(self, X, y):
        """ Fit method for AFE

        It sets up the primitive set, toolbox, creates surrogate dataframe, creates initial population and starts the
        evolution. Future Work: checking OneHotEncoding, datetime etc


        :param X: numpy ndarray input matrix [n_samples, n_features]
        :param y: numpy ndarray target values
        :return: None
        """
        self._X = reshape_numpy(X)
        self._y = reshape_numpy(y)
        self._X_train, self._X_val, self._y_train, self._y_val = \
            train_test_split(self._X, self._y, test_size=0.2, random_state=self._random_state)
        self._feature_count = self._X.shape[1]
        self._setup_pset()
        self._setup_toolbox()
        self._pop = self._toolbox.population(self._pop_size)
        self._evolve()

    def _evaluate(self, individual):
        """ Evaluate the individual during evolution

        Given an estimator and a scorer, evaluate the individual and append the individual config to surrogate training
        dataframe. Future Work: Handle these exceptions.

        :param individual:
        :return: a tuple of score
        """
        input_matrix = deepcopy(self._X_train)
        target = self._y_train
        pipeline = self._compile_to_sklearn(individual=individual)
        score = 0
        try:
            pipeline.fit(input_matrix, target)
            y_pred = pipeline.predict(self._X_val)
            score = self._scorer(self._y_val, y_pred)
            return score, individual.height
        except Exception as e:
            return score, individual.height

    @abc.abstractmethod
    def predict(self, X=None, y=None):
        """ abstract method,  Returns a pipeline that yields the best score for the given estimator and scorer

        :param X: numpy ndarray input matrix [n_samples, n_features]
        :param y: numpy ndarray target values
        :return: a tuple of sklearn pipeline and a instance of estimator
        """
        pass

    def _solution_to_file(self, f_name='solution'):
        """ Future Work: Export best pipeline and the corresponding code to the file

        :param f_name: string, file name
        :return: None
        """
        pass

    def _initialise_replay(self):
        """ Initialise the replay memory for DQN

        :return: None
        """
        self._replay = Replay_Memory(random_state=self._random_state)
