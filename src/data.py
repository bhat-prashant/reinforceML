import logging

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer


def impute(X, y=None):
    imputer = SimpleImputer()
    X = imputer.fit_transform(X)
    return X


class OneHotEncoder(BaseEstimator, TransformerMixin):
    def _digest_shape(self, X):
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

    def fit(self, X, y=None):
        self.classes_ = list(sorted(set(self._digest_shape(X))))
        return self

    def transform(self, X, y=None):
        M = self._digest_shape(X)
        M = np.array(M)
        R = [M == c for c in self.classes_]
        R = np.column_stack(R)
        return R


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key, row_space=True, as_matrix=True):
        self.key = key
        self.row_space = row_space
        self.as_matrix = as_matrix

    def fit(self, X, y=None):
        return self

    def transform(self, data_matrix):
        if self.row_space:
            R = data_matrix[:, [self.key]]  # eg numpy array
        else:
            R = data_matrix[[self.key]]  # eg pandas dataframe

        R = np.array(R)

        if not self.as_matrix:
            R = R[:, 0]

        return R


# TODO - Build intelligent transformations. Add more to the list in the future.
def get_transformers():
    transformers = {
        'add': np.add,
        'subtract': np.subtract,
        'multiply': np.multiply,
        'division': np.divide,
        'log': np.log,
        'square': np.square,
        'square_root': np.sqrt
    }
    return transformers


def create_chromosomes(X, y=None, original=True, transform=True, transformers=None):
    chromosomes = []
    X_real = []
    for i in range(X.shape[1]):
        X_real.append(X[:, i])
    if original:
        chromosomes.extend(X_real)
    if transform:
        if not isinstance(transformers, dict):
            logging.error("Unknown transformer list. Expected type \'dict\', got instead ", type(transformers))
            transformers = get_transformers()
        for chrome in X_real:
            for i in transformers:
                chromosomes.extend(transformers[i](chrome))
    return chromosomes
