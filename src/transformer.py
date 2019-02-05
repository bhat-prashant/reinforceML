#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np




class Output_Array(object):
    """Output data type of pipelines."""
    pass


class ARGType(object):
    """Base class for parameter specifications."""
    pass


class BaseTransformer(object):
    # class variables
    root = False
    arg_types = None


# Base class for numpy operators
class BaseNumpyTransformer(BaseEstimator, TransformerMixin):
    def fit(self, *args):
        pass

    def transform(self, *args, **kwargs):
        return self.fit_transform(*args, **kwargs)

    def fit_transform(self, X, y=None, **kwargs):
        # implement numpy transformation here
        pass


# add operation using numpy.add
class AddNumpy(BaseNumpyTransformer):
    def fit_transform(self, X, y=None, **kwargs):
        try:
            i_0 = kwargs['index0']
            i_1 = kwargs['index1']
            transformed_X = np.add(X[:, i_0], X[:, i_1])
            return np.append(X, transformed_X, axis=1)
        except KeyError as e:
            return X


# subtract operation using numpy.subtract
class SubtractNumpy(BaseNumpyTransformer):
    def fit_transform(self, X, y=None, **kwargs):
        try:
            i_0 = kwargs['index0']
            i_1 = kwargs['index1']
            transformed_X = np.subtract(X[:, i_0], X[:, i_1])
            return np.append(X, transformed_X, axis=1)
        except KeyError as e:
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








