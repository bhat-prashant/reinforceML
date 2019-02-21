#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.decomposition import PCA

import numpy as np
from utils_ import reshape_numpy
from copy import deepcopy

# output class for scalers
class ScaledArray(object):
    pass

# output class for selectors
class SelectedArray(object):
    pass

# output class for extractors such as PCA
class ExtractedArray(object):
    pass

class ARGType(object):
    """Base class for parameter specifications."""
    pass


class BaseTransformer(object):
    # class variables
    arg_types = None


# Wrapper class for non sklearn estimators and estimators whose parameters have to be tweaked dynamically(Ex: PCA)
class BaseReinforceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.args = kwargs

    def fit(self, *args):
        return self



# add operation using numpy.add
class AddReinforce(BaseReinforceTransformer):
    def transform(self, X, y=None):
        i_0 = self.args['index0']
        i_1 = self.args['index1']
        transformed_X = np.add(X[:, i_0], X[:, i_1])
        transformed_X = reshape_numpy(transformed_X)
        return np.append(X, transformed_X, axis=1)


# subtract operation using numpy.subtract
class SubtractReinforce(BaseReinforceTransformer):
    def transform(self, X, y=None):
        i_0 = self.args['index0']
        i_1 = self.args['index1']
        transformed_X = np.subtract(X[:, i_0], X[:, i_1])
        transformed_X = reshape_numpy(transformed_X)
        return np.append(X, transformed_X, axis=1)


# multiply operation using numpy.multiply
class MultiplyReinforce(BaseReinforceTransformer):
    def transform(self, X, y=None):
        i_0 = self.args['index0']
        i_1 = self.args['index1']
        transformed_X = np.multiply(X[:, i_0], X[:, i_1])
        transformed_X = reshape_numpy(transformed_X)
        return np.append(X, transformed_X, axis=1)

# divide operation using numpy.divide
class DivideReinforce(BaseReinforceTransformer):
    def transform(self, X, y=None):
        i_0 = self.args['index0']
        i_1 = self.args['index1']
        divisor = deepcopy(X[:, i_1])
        divisor[divisor < 1] = 1
        transformed_X = np.divide(X[:, i_0], divisor)
        transformed_X = reshape_numpy(transformed_X)
        return np.append(X, transformed_X, axis=1)

# log operation using numpy.log
class LogReinforce(BaseReinforceTransformer):
    def transform(self, X, y=None):
        i_0 = self.args['index0']
        temp = deepcopy(X[:, i_0])
        # log operation is applicable only on numbers greater than zero!
        temp[temp < 0.0001] = 0.0001
        transformed_X = np.log(temp)
        transformed_X = reshape_numpy(transformed_X)
        return np.append(X, transformed_X, axis=1)

# KBinsDiscretizer
class KBinsDiscretizerReinforce(BaseReinforceTransformer):
    def transform(self, X, y=None):
        i_0 = self.args['index0']
        strategy = self.args['strategy']
        n_bins = self.args['n_bins']
        transformed_X = KBinsDiscretizer(n_bins=n_bins, strategy=strategy).fit_transform(X[:, i_0:i_0+1])
        transformed_X.indices = reshape_numpy(transformed_X.indices)
        # transformed column is appended to input_matrix
        return np.append(X, transformed_X.indices, axis=1)

# PCA - This is a workaround when n_components is greater than no.of features due to feature selection before.
class PCAReinforce(BaseReinforceTransformer):
    def transform(self, X, y=None):
        n_components = self.args['n_components']
        whiten = self.args['whiten']
        svd_solver = self.args['svd_solver']
        if n_components >= X.shape[1]:
            n_components = X.shape[1]-1
        transformed_X = PCA(n_components=n_components, whiten=whiten, svd_solver=svd_solver).fit_transform(X, y)
        return transformed_X


# Does nothing!
class EmptyTransformer(BaseReinforceTransformer):
    def transform(self, X, y=None, **kwargs):
        return X


# create argument class dynamically
def ArgClassGenerator(transformer_name, arg_name, range, BaseClass=ARGType):
    return type(transformer_name + '_' + arg_name, (BaseClass,), {'values': range, 'name':arg_name})


# create transformer class dynamically
def TransformerClassGenerator(name, transformerdict, BaseClass=BaseTransformer, ArgClass=ARGType):
    arg_types = []
    # build arg classes
    for arg in transformerdict['params']:
        arg_types.append(ArgClassGenerator(name, arg, transformerdict['params'][arg], ArgClass))
    # build class attributes
    profile = {'transformer': transformerdict['transformer'], 'arg_types': arg_types }
    return type(name, (BaseClass,), profile)








