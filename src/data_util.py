
import logging
import numpy as np



def validate_inputs(X, y):
    valid = True
    if isinstance(X, np.ndarray) or isinstance(y, np.ndarray):
        pass
        # Future Work: - preprocessing such as OneHotEncoding, Imputing, Scaling etc.
    else:
        valid = False
        logging.error('Expected \'numpy.ndarray\' as inputs, Instead got {} and {}', type(X), type(y) )
    return valid








































