
import numpy as np
import logging
from sklearn.base import BaseEstimator, TransformerMixin


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

    def transform(self, *args):
        if len(args) == self.param_count:
            return self.transformer(*args)
        else:
            logging.error("Error! Expecting %d parameters, instead got %d".format(self.param_count, len(*args)))


class UnaryTransformer(BaseTransformer):
    def __init__(self, name, transformer, param_count=1, ):
        super(UnaryTransformer, self).__init__(transformer, param_count, name)


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
    # transformers['log'] = UnaryTransformer(transformer=np.log)
    transformers['square'] = UnaryTransformer(name='square', transformer=np.square)
    transformers['square_root'] = UnaryTransformer(name='square_root', transformer=np.sqrt)
    return transformers
