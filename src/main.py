#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"

import abc
import random
import warnings
import scipy
import numpy as np
import pandas as pd
from deap import gp, creator, base, tools
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from gp_ import grow_individual, mutate, cxOnePoint, eaMuPlusLambda
from lookup import TransformerLookUp
from transformer import TransformerClassGenerator, ScaledArray, SelectedArray, ExtractedArray, ClassifiedArray

random.seed(10)
np.random.seed(10)


class BaseReinforceML(BaseEstimator, TransformerMixin, metaclass=abc.ABCMeta):

    def __init__(self, estimator, reinforce_learner, generation, pop_size, mutation_rate,
                 crossover_rate, scorer, inputArray, outputArray, trans_types):
        self._generation = generation
        self._pop_size = pop_size
        self._mutation_rate = mutation_rate
        self._crossover_rate = crossover_rate
        self._estimator = estimator
        self._reinforce_learner = reinforce_learner
        self._scorer = scorer
        self._inputArray = inputArray
        self._outputArray = outputArray
        self._trans_types = trans_types
        self._feature_count = None
        self._initial_score = None
        self._pop = None
        self._toolbox = None
        self._X = None
        self._y = None
        self._pset = None
        self._pandas_columns = None

    # create typed PrimitiveSet
    def _setup_pset(self):
        if self._inputArray is not None and self._outputArray is not None and isinstance(self._trans_types, list):
            self._pset = gp.PrimitiveSetTyped('MAIN', self._inputArray, self._outputArray)
            self._pset.renameArguments(ARG0='input_matrix')
            lookup = TransformerLookUp(self._feature_count)
            self._pandas_columns = []  # input features for RL training
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
                            self._pandas_columns.append(arg_name)

    def _setup_toolbox(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            creator.create('FitnessMax', base.Fitness, weights=(1.0,))
            creator.create('Individual', gp.PrimitiveTree, fitness=creator.FitnessMax, statistics=dict)
        self._toolbox = base.Toolbox()
        self._toolbox.register('expr', grow_individual, pset=self._pset, trans_types=self._trans_types, min_=1, max_=8)
        self._toolbox.register('individual', tools.initIterate, creator.Individual, self._toolbox.expr)
        self._toolbox.register('population', tools.initRepeat, list, self._toolbox.individual)
        self._toolbox.register('evaluate', self._evaluate)
        self._toolbox.register('select', tools.selBest)
        self._toolbox.register('mate', cxOnePoint)
        self._toolbox.register('mutate', mutate, self._pset)
        self._toolbox.register('train_RL', self._train_RL)

    # compile individual into a sklearn pipeline
    # Note: pipeline is appended with self._estimator as the final step
    def _compile_to_sklearn(self, individual):
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
        # add a estimator to the pipeline
        pipeline.append(self._estimator)
        pipe = make_pipeline(*pipeline)
        return pipe

    # export best sklearn-pipelines to a file
    def _solution_to_file(self, f_name='solution'):
        pass

    # fit and predict (input, target) for given pipeline
    # Note: pipeline is appended with self._estimator as the final step
    @abc.abstractmethod
    def _evaluate(self, individual):
        pass

    def _evolve(self):
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

    # From a list of index tuples, create pandas dataframe
    def _create_dataframe(self):
        # target column for RL training
        self._pandas_columns.append('reward')
        self._rl_dataframe = pd.DataFrame(columns=self._pandas_columns)

    # Train reinforcement learning
    # Future Work: Use GridSearchCV instead of a bare estimator
    def _train_RL(self):
        self._rl_dataframe = self._rl_dataframe.drop_duplicates()
        self._rl_dataframe = self._rl_dataframe.reset_index(drop=True)
        X = self._rl_dataframe.iloc[:, :-1].values
        y = self._rl_dataframe.iloc[:, -1].values
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=10)
        self._reinforce_learner.fit(X_tr, y_tr)
        y_pr = self._reinforce_learner.predict(X_te)
        print("RL score : ", scipy.stats.pearsonr(y_te, y_pr))

    # Predict reward from reinforcement learning
    def _predict_RL(self, dataframe):
        X = dataframe.iloc[:, :-1].values
        ypred = self._reinforce_learner.predict(X)
        print("Predicted : ", ypred)

    @abc.abstractmethod
    def fit(self, X, y):
        pass

    @abc.abstractmethod
    def predict(self):
        pass
