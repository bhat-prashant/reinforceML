#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer


from decorator_ import unary_decorator
from constants import *

class BaseTransformer:
    __slots__ = ['transformer', 'param_count', 'name']

    def __init__(self, transformer, param_count, name):
        self.transformer = transformer
        # number of parameters in a mathematical transformer, ex. add() has 2
        self.param_count = param_count
        self.name = name
        pass

    def set_transformer(self, transformer):
        if transformer is not None:
            self.transformer = transformer

    def get_transformer(self):
        return self.transformer

    def get_param_count(self):
        return self.param_count

    def set_param_count(self, param_count=None):
        self.param_count = param_count



class UnaryTransformer(BaseTransformer):
    def __init__(self, name, transformer, param_count=1, ):
        super(UnaryTransformer, self).__init__(transformer, param_count, name)

    # feat_imp is set default to -1 for newly transformed features. Their importance will be updated later during evaluation
    # indices and node_names should be iterables
    # This method makes changes to passed 'individual' and hence returns None
    @unary_decorator
    def transform(self, individual, index, feat_imp=-1):
        try:
            if individual.data.ndim == 1:
                data = self.transformer(individual.data)
            else:
                data = self.transformer(individual.data[:, index:(index+1)])
            node_name = individual.meta_data[index][N_NAME]
            new_node_name = self.name + '(' + node_name + ')'
            individual.meta_data[index][A_GRAPH].add_node(str(new_node_name))
            individual.meta_data[index][A_GRAPH].add_edge(node_name, new_node_name, transformer=self.name)
            individual.meta_data[index][N_NAME] = new_node_name
            individual.meta_data[index][F_IMP] = feat_imp
            return data
        except Exception as e:
            print('Unknown error while transforming a feature !', e)


class BinaryTransformer(BaseTransformer):
    def __init__(self, name, transformer, param_count=2):
        super(BinaryTransformer, self).__init__(transformer, param_count, name)


class HigherOrderTransformer(BaseTransformer):
    def __init__(self, name, transformer, param_count=None):
        super(HigherOrderTransformer, self).__init__(transformer, param_count, name)


class OneHotEncoder(BaseEstimator, TransformerMixin):
    def _digest_shape(X):
        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                M = X
            elif X.ndim == 2:
                M = X[:, 0]
            else:
                raise ValueError('One hot encoder does not work with nd, n>2 data')
        elif isinstance(X, list):
            if isinstance(X[0], list):
                M = [x[0] for x in X]
            else:
                M = X
        return M

    def fit(self, X):
        self.classes_ = list(sorted(set(self._digest_shape(X))))
        return self

    def transform(self, X):
        M = self._digest_shape(X)
        M = np.array(M)
        R = [M == c for c in self.classes_]
        R = np.column_stack(R)
        return R

# Future Work : Add all possible transformers.
def get_transformers():
    transformers = dict()
    # transformers['add'] = BinaryTransformer(transformer=np.add)
    # transformers['subtract'] = BinaryTransformer(transformer=np.subtract)
    # transformers['multiply'] = BinaryTransformer(transformer=np.multiply)
    # transformers['division'] = BinaryTransformer(transformer=np.divide)
    transformers[LOG] = UnaryTransformer(name=LOG, transformer=np.log)
    transformers[SQR] = UnaryTransformer(name=SQR, transformer=np.square)
    transformers[SQRT] = UnaryTransformer(name=SQRT, transformer=np.sqrt)
    transformers[KBD] = UnaryTransformer(name=KBD, transformer=KBinsDiscretizer().fit_transform)
    return transformers
