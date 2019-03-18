#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"

import abc
import random
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import scipy
from deap import gp, creator, base, tools
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from gp_ import grow_individual, mutate, cxOnePoint, eaMuPlusLambda
from lookup import TransformerLookUp
from transformer import TransformerClassGenerator, ScaledArray, SelectedArray, ExtractedArray
from utils_ import reshape_numpy

random.seed(10)
np.random.seed(10)


class BaseReinforceML(BaseEstimator, TransformerMixin, metaclass=abc.ABCMeta):

    def __init__(self, estimator, reinforce_learner, generation=5, pop_size=None, mutation_rate=0.3,
                 crossover_rate=0.7, scorer=accuracy_score):
        self._generation = generation
        self._pop_size = pop_size
        self._mutation_rate = mutation_rate
        self._crossover_rate = crossover_rate
        self._estimator = estimator
        self._reinforce_learner = reinforce_learner
        self._scorer = scorer
        self._feature_count = None
        self._initial_score = None
        self._pop = None
        self._toolbox = None
        self._X = None
        self._y = None
        self._pset = None
        self._pandas_columns = None

    # create typed PrimitiveSet
    def _setup_pset(self, outputArray=None, inputArray=None, trans_types=None):
        if inputArray is None:
            inputArray = [np.ndarray]
        self._pset = gp.PrimitiveSetTyped('MAIN', inputArray, outputArray)
        self._pset.renameArguments(ARG0='input_matrix')
        trans_types = ['unary', 'scaler', 'selector', 'extractor']
        lookup = TransformerLookUp(self._feature_count)
        self._pandas_columns = []  # input features for RL training
        for type_ in trans_types:
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
                    self._pset.addPrimitive(transformer, [ScaledArray] + transformer.arg_types, SelectedArray)
                elif type_ == 'extractor':
                    self._pset.addPrimitive(transformer, [SelectedArray] + transformer.arg_types, ExtractedArray)

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
        self._toolbox.register('expr', grow_individual, pset=self._pset, min_=1, max_=8)
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

    # Initialization steps
    def _fit_init(self):
        self._feature_count = self._X.shape[1]
        # initial accuracy on the given dataset
        self._estimator.fit(self._X_train, self._y_train)
        y_pred = self._estimator.predict(self._X_val)
        self._initial_score = roc_auc_score(self._y_val, y_pred)
        print('Initial Best score : ', self._initial_score)
        # setup toolbox for evolution
        self._setup_pset()
        # create dataframe for RL training
        self._create_dataframe()
        # create population
        self._setup_toolbox()
        # Future Work: Some of these ind are taking too much time. Inspect why!
        # create population
        self._pop = self._toolbox.population(self._pop_size)
        # start evolution
        self._evolve()

    def fit(self, X, y):
        # Future Work: checking OneHotEncoding, datetime etc
        self._X = reshape_numpy(X)
        self._y = reshape_numpy(y)
        self._X_train, self._X_val, self._y_train, self._y_val = train_test_split(self._X, self._y, test_size=0.2,
                                                                                  random_state=10)
        print('Dataset target distribution %s' % Counter(y))
        self._fit_init()
        print(self._hof[0])
        return self._compile_to_sklearn(self._hof[0]), self._estimator
