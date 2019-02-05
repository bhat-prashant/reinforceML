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
from copy import deepcopy

from gp_ import grow_individual
from utils_ import operator_precheck
from lookup import TransformerLookUp
from metrics_ import fitness_score
from transformer import TransformerClassGenerator, ScaledArray, SelectedArray, ExtractedArray
from sklearn.base import BaseEstimator, TransformerMixin


class BaseFeatureEngineer(BaseEstimator, TransformerMixin):

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
        self._pset = gp.PrimitiveSetTyped('MAIN', [np.ndarray], ExtractedArray)
        self._pset.renameArguments(ARG0='input_matrix')
        trans_types = ['unary', 'scaler', 'selector', 'extractor']
        lookup = TransformerLookUp(self._feature_count)
        for type_ in trans_types:
            trans_lookup = lookup.get_lookup(type_)

            # add transformers as primitives
            for key in trans_lookup:
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


    def _setup_toolbox(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            creator.create('FitnessMulti', base.Fitness, weights=(1.0,))
            creator.create('Individual', gp.PrimitiveTree, fitness=creator.FitnessMulti, statistics=dict)

        self._toolbox = base.Toolbox()
        self._toolbox.register('expr', grow_individual, pset=self._pset, min_=1, max_=8)
        self._toolbox.register('individual', tools.initIterate, creator.Individual, self._toolbox.expr)
        self._toolbox.register('population', tools.initRepeat, list, self._toolbox.individual)
        self._toolbox.register('evaluate', self._evaluate_individual)
        self._toolbox.register('select', tools.selNSGA2)
        self._toolbox.register('mate', tools.cxOrdered)
        # self._toolbox.register('expr_mut', self._gen_grow_safe, min_=1, max_=4)
        # self._toolbox.register('mutate', self._random_mutation_operator)



    def _evaluate_individual(self, individual):
        input_matrix = deepcopy(self._X)
        target = self._y
        height  = individual.height - 1 # start from first primitive
        arg_pos = individual.height + 1 # start from first argument after input_matrix for the above primitive

        # for every primitive transformer, do
        while height >= 0:
            # get primitive class
            trans_class = self._pset.context[individual[height].name]
            args = {}

            # get corresponding arguments
            for arg_type in trans_class.arg_types:
                args[arg_type.name] = self._pset.context[individual[arg_pos].name]
                arg_pos = arg_pos + 1

            # apply transformation based on package name
            try:
                operator = trans_class.transformer(**args)
                operator, input_matrix = operator_precheck(operator, input_matrix, **args)
                input_matrix = operator.fit_transform(input_matrix, y=self._y)
            except Exception as e:
                print(e)
            # Start from most basic primitive and move up the tree towards root / universal primitive
            height = height - 1

        # evaluate individual
        try:
            fitness, _ = fitness_score(input_matrix, self._y)
        except:
            fitness = 0
        print(fitness, ' : ', individual)





    # Initialization steps
    def _fit_init(self):
        self._feature_count = self._X.shape[1]

        # initial accuracy on the given dataset
        self._initial_score, _ = fitness_score(self._X, self._y)
        print('Initial Best score : ', self._initial_score)

        # setup toolbox for evolution
        self._setup_pset()

        # create population
        self._setup_toolbox()
        self._pop = self._toolbox.population(100)

        for ind in self._pop:
            self._evaluate_individual(individual=ind)




    def fit(self, X, y):
        # Future Work: checking OneHotEncoding, datetime etc
        self._X = X
        self._y = y
        print('Dataset target distribution %s' % Counter(y))
        self._fit_init()
