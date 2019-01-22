'''
TODO --
1. Mutate
2. Cross Over
3. Take special note for handling numpy arrays ( https://deap.readthedocs.io/en/master/tutorials/advanced/numpy.html )
4. Reinforcement Learning

'''
import numpy as np

from base import BaseTransformer


class UnaryTransformer(BaseTransformer):
    def __init__(self, transformer=None, param_count=1):
        super(UnaryTransformer, self).__init__(transformer, param_count)


class BinaryTransformer(BaseTransformer):
    def __init__(self, transformer=None, param_count=2):
        super(BinaryTransformer, self).__init__(transformer, param_count)


class HigherOrderTransformer(BaseTransformer):
    def __init__(self, transformer=None, param_count=None):
        super(HigherOrderTransformer, self).__init__(transformer, param_count)


def get_transformers():
    transformers = dict()
    # transformers['add'] = BinaryTransformer(transformer=np.add)
    # transformers['subtract'] = BinaryTransformer(transformer=np.subtract)
    # transformers['multiply'] = BinaryTransformer(transformer=np.multiply)
    # transformers['division'] = BinaryTransformer(transformer=np.divide)
    # transformers['log'] = UnaryTransformer(transformer=np.log)
    transformers['square'] = UnaryTransformer(transformer=np.square)
    transformers['square_root'] = UnaryTransformer(transformer=np.sqrt)
    return transformers
