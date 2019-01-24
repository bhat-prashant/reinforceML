
import logging

import numpy as np
from transformer import UnaryTransformer, BinaryTransformer, HigherOrderTransformer, get_transformers



def validate_inputs(X, y):
    valid = True
    if isinstance(X, np.ndarray) or isinstance(y, np.ndarray):
        pass
        # Future Work: - preprocessing such as OneHotEncoding, Imputing, Scaling etc.
    else:
        valid = False
        logging.error('Expected \'numpy.ndarray\' as inputs, Instead got {} and {}', type(X), type(y) )
    return valid




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
                if isinstance(trans, UnaryTransformer):  # Future Work: - binary and higher order transform
                    chromosomes.append(trans.transform(chrome))
                elif isinstance(trans, BinaryTransformer):
                    pass
                elif isinstance(trans, HigherOrderTransformer):
                    pass
                else:
                    logging.error("Unknown transformer type : ", type(trans))
    return chromosomes






































