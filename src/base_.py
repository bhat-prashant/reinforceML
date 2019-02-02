#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"

import random
import warnings
from collections import Counter

import numpy as np
from deap import creator, base, tools
from deap import gp
from sklearn.metrics import accuracy_score

from gp_ import grow_individual
from lookup import get_lookup
from metrics_ import fitness_score
from transformer import TransformerClassGenerator, Output_Array


class BaseFeatureEngineer:

    def __init__(self, generation=5, pop_size=None, mutation_rate=0.3,
                 crossover_rate=0.7, scorer=accuracy_score, upsample=False,
                 downsample=False, random_state=None, verbosity=0, subsample=None, append_original=True):
        self._generation = generation
        self._pop_size = pop_size
        self._mutation_rate = mutation_rate
        self._crossover_rate = crossover_rate
        self._scorer = scorer
        self._feature_count = None
        self._initial_score = None
        self._pop = None
        self._toolbox = None
        self._X = None
        self._y = None

    # create typed PrimitiveSet
    def _setup_pset(self):
        random.seed(10)
        np.random.seed(10)
        self._pset = gp.PrimitiveSetTyped('MAIN', [np.ndarray], Output_Array)
        self._pset.renameArguments(ARG0='input_matrix')
        trans_lookup = get_lookup()

        # add transformers as primitives
        for key in trans_lookup:
            transformer = TransformerClassGenerator(key, trans_lookup[key])
            if transformer.root:
                self._pset.addPrimitive(transformer, [np.ndarray] + transformer.arg_types, Output_Array)
            else:
                self._pset.addPrimitive(transformer, [np.ndarray] + transformer.arg_types, np.ndarray)

            # add transformer arguments as terminal
            # arg_types is a list
            for arg in transformer.arg_types:
                values = list(arg.values)
                for val in values:
                    arg_name = arg.__name__ + "=" + str(val)
                    self._pset.addTerminal(val, arg, name=arg_name)


    def _setup_toolbox(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            creator.create('FitnessMulti', base.Fitness, weights=(1.0,))
            creator.create('Individual', gp.PrimitiveTree, fitness=creator.FitnessMulti, statistics=dict)

        self._toolbox = base.Toolbox()
        self._toolbox.register('expr', grow_individual, pset=self._pset, min_=1, max_=3)
        self._toolbox.register('individual', tools.initIterate, creator.Individual, self._toolbox.expr)
        self._toolbox.register('population', tools.initRepeat, list, self._toolbox.individual)
        # self._toolbox.register('compile', self._compile_to_sklearn)
        # self._toolbox.register('select', tools.selNSGA2)
        # self._toolbox.register('mate', self._mate_operator)
        # self._toolbox.register('expr_mut', self._gen_grow_safe, min_=1, max_=4)
        # self._toolbox.register('mutate', self._random_mutation_operator)


    # Initialization steps
    def _fit_init(self):
        self._feature_count = self._X.shape[1]

        # initial accuracy on the given dataset
        self._initial_score, _ = fitness_score(self._X, self._y)

        # setup toolbox for evolution
        self._setup_pset()

        # create population
        # self._pop = self._toolbox.population()



    def fit(self, X, y):
        # Future Work: checking OneHotEncoding, datetime etc
        self._X = X
        self._y = y
        print('Dataset target distribution %s' % Counter(y))
        self._fit_init()
