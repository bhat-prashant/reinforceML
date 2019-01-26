
import numpy as np
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



class UnaryTransformer(BaseTransformer):
    def __init__(self, name, transformer, param_count=1, ):
        super(UnaryTransformer, self).__init__(transformer, param_count, name)

    # feat_imp is set default to -1 for newly transformed features. Their importance will be updated later during evaluation
    # indices and node_names should be iterables
    # This method makes changes to passed 'individual' and hence returns None
    def transform(self, individual, indices, node_names, feat_imp=-1):
        for index, node_name in zip(indices, node_names):
            if individual.data.ndim == 1:
                individual.data = self.transformer(individual.data)
            else:
                individual.data = self.transformer(individual.data[:, index])
            new_node_name = self.name + '(' + node_name + ')'
            individual.meta_data[index]['ancestor_graph'].add_node(str(new_node_name))
            individual.meta_data[index]['ancestor_graph'].add_edge(node_name, new_node_name, transformer=self.name)
            individual.meta_data[index]['node_name'] = new_node_name
            individual.meta_data[index]['feature_importance'] = feat_imp



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
    # transformers['squareuare'] = UnaryTransformer(name='square', transformer=np.square)
    transformers['square_root'] = UnaryTransformer(name='square_root', transformer=np.sqrt)
    return transformers
