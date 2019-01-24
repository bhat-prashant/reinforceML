import numpy as np
import random


# crossover / mate function for two individuals
def mate(individual1, individual2, relevacne=0.25):
    # Retain features with importance more than the relevance and merge two individuals.
    # Future Work : come up with 'intelligent' mating technique. Mating between individuals should be based on the 'attraction'
    # i.e Good 'looking' individuals should mate with other good 'looking' (accuracy) individuals
    i1 = [i for i in range(individual1.feature_importance) if individual1.feature_importance[i] > relevacne]
    individual1.data = individual1.data[:, i1]
    individual1.features = [individual1.features[j] for j in i1]
    i2 = [i for i in range(individual2.feature_importance) if individual2.feature_importance[i] > relevacne]
    individual2.data = individual2.data[:, i2]
    individual2.features = [individual2.features[j] for j in i2]
    individual1.merge(individual2)
    return individual1


# Future Work: Reinforcement Learning
def mutate(individual, transformers):
    key = random.choice(list(transformers.keys()))
    # Future Work : Decorators for pre and post sanity check
    # Example : Features are not compatible, Go out of bound after squaring many times etc.
    individual.data = transformers[key].transform(individual.data)
    for feat in individual.features:
        feat.transformer_list.extend(transformers[key].name)



class Individual:

    __slots__ = ['fitness', 'data', 'feature_importance', 'features']

    def __init__(self, data):
        self.fitness = 0
        self.data = data
        self.feature_importance = []
        self.features = []
        self.extract_features()

    def extract_features(self):
        if self.data.ndim == 1:
            self.data = np.reshape(self.data, (self.data.shape[0], 1))
        for i in range(self.data.shape[1]):
            feat = Feature(i)
            self.features.append(feat)
            self.feature_importance.append(0)

    def merge(self, individual):
        self.fitness = 0
        self.data = np.append(self.data, individual.data, axis=1)
        self.feature_importance.extend(individual.feature_importance)
        self.features.extend(individual.features)
        for idx, feat in enumerate(self.features):
            feat.index = idx



class Feature:

    __slots__ = ['transformer_list', 'index']

    # Future Work : Add feature's meta information such as 'datetime', 'categorical', 'timeseries' etc
    def __init__(self, index):
        self.transformer_list = []
        self.index = index