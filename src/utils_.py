#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"

import numpy as np
from sklearn.preprocessing import KBinsDiscretizer


def reshape_data(individual):
    individual.data = reshape_numpy(individual.data)


def reshape_numpy(ndarray):
    if ndarray.ndim == 1:
        ndarray = np.reshape(ndarray, (ndarray.shape[0], 1))
    return ndarray

# sklearn operator manipulation
def _apply_sklearn_operator(operator, in_matrix, indices):
    if isinstance(operator, KBinsDiscretizer):
        # Future Work: Take care of convergence warning, User warning
        try:
            data = operator.fit_transform(in_matrix[:, indices[0]:indices[0] + 1])
            return data.indices
        except Exception as e:
            # Future Work: Remove operator from the individual as it cannot be applied
            return in_matrix[:, indices[0]]


# numpy operator manipulation
# Future Work: Handle run time warnings (especially for log transformation)
def _apply_numpy_operator(operator, in_matrix, indices):
    try:
        # Generalized for any number of inputs (i.e unary, binary or higher order numpy transformer)
        data = in_matrix[:, indices]
        data = operator(*np.split(data, data.shape[1], axis=1))
        if np.isnan(data).any() or np.isinf(data).any():
            raise Exception
        return data
    except Exception as e:
        # Future Work: Remove operator from the individual as it cannot be applied
        return in_matrix[:, indices]








