#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
from utils_ import reshape_numpy


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


# Base class for numpy operators
class BaseNumpyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.args = kwargs

    def fit(self, *args):
        return self



# add operation using numpy.add
class AddReinforce(BaseNumpyTransformer):
    def transform(self, X, y=None):
        i_0 = self.args['index0']
        i_1 = self.args['index1']
        transformed_X = np.add(X[:, i_0], X[:, i_1])
        transformed_X = reshape_numpy(transformed_X)
        return np.append(X, transformed_X, axis=1)


# subtract operation using numpy.subtract
class SubtractReinforce(BaseNumpyTransformer):
    def transform(self, X, y=None):
        i_0 = self.args['index0']
        i_1 = self.args['index1']
        transformed_X = np.subtract(X[:, i_0], X[:, i_1])
        transformed_X = reshape_numpy(transformed_X)
        return np.append(X, transformed_X, axis=1)


# KBinsDiscretizer
class KBinsDiscretizerReinforce(BaseNumpyTransformer):
    def transform(self, X, y=None):
        i_0 = self.args['index0']
        strategy = self.args['strategy']
        n_bins = self.args['n_bins']
        transformed_X = KBinsDiscretizer(n_bins=n_bins, strategy=strategy).fit_transform(X[:, i_0:i_0+1])
        transformed_X.indices = reshape_numpy(transformed_X.indices)
        return np.append(X, transformed_X.indices, axis=1)



# Does nothing!
class EmptyTransformer(BaseNumpyTransformer):
    def transform(self, X, y=None, **kwargs):
        return X


# create argument class dynamically
def ArgClassGenerator(transformer_name, arg_name, range, BaseClass=ARGType):
    return type(transformer_name + '_' + arg_name, (BaseClass,), {'values': range, 'name':arg_name})


# create transformer class dynamically
def TransformerClassGenerator(name, transformerdict, BaseClass=BaseTransformer, ArgClass=ARGType):
    arg_types = []
    for arg in transformerdict['params']:
        arg_types.append(ArgClassGenerator(name, arg, transformerdict['params'][arg], ArgClass))
    # build class attributes
    profile = {'transformer': transformerdict['transformer'], 'arg_types': arg_types }
    return type(name, (BaseClass,), profile)








