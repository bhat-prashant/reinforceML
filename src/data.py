
import logging

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

from transformer import UnaryTransformer, BinaryTransformer, HigherOrderTransformer, get_transformers


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


def impute(X):
    imputer = SimpleImputer()
    X = imputer.fit_transform(X)
    return X



def create_chromosomes(X, y=None, original=True, transform=True, transformers=None):
    chromosomes = []
    X_real = []
    for i in range(X.shape[1]):
        X_real.append(X[:, i])
    if original:
        chromosomes.extend(X_real)
    if transform:
        if transformers is None:
            transformers = get_transformers()
        for chrome in X_real:
            for trans in transformers.values():
                if isinstance(trans, UnaryTransformer):  # Todo - binary and higher order transform
                    chromosomes.append(trans.transform(chrome))
                elif isinstance(trans, BinaryTransformer):
                    pass
                elif isinstance(trans, HigherOrderTransformer):
                    pass
                else:
                    logging.error("Unknown transformer type : ", type(trans))
    return chromosomes
