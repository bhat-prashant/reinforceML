import random

import numpy as np


# crossover / mate function for two individuals
def mate(individual1, individual2, relevacne=0.25):
    # Retain features with importance more than the relevance and merge two individuals.
    # Future Work : come up with 'intelligent' mating technique. Mating between individuals should be based on the 'attraction'
    # i.e Good 'looking' individuals should mate with other good 'looking' (accuracy) individuals
    return individual1


# Future Work: Reinforcement Learning
def mutate(transformers, individual):
    key = random.choice(list(transformers.keys()))
    # Future Work : Decorators for pre and post sanity check
    # Example : Features are not compatible, Go out of bound after squaring many times etc.
    if transformers[key].param_count == 1:
        # Future Work: Do not apply UnaryTransformation on all features. select features 'intelligently'
        for index in range(len(individual.meta_data)):
            node_name = individual.meta_data[index]['node_name']
            individual.data = transformers[key].transform(individual.data)
            new_node_name = transformers[key].name + '(' + node_name + ')'
            G = individual.meta_data[index]['ancestor_graph']
            G.add_node(str(new_node_name))
            G.add_edge(node_name, new_node_name, transformer=transformers[key].name)
            # Update every features meta_data after applying transformation
            individual.meta_data[index]['node_name'] = new_node_name
            individual.meta_data[index]['feature_importance'] = -1
            individual.meta_data[index]['ancestor_graph'] = G
    return individual



# select best individuals after cross over and mutation
# Future Work : 'Intelligently' make selection. / create exponential decay in selection
# Now, Top 90% and random 5% (lucky few)
def select(pop, top=0.80, lucky=0.05):
    pop.sort(key=lambda x: x.fitness, reverse=True)
    top_index = int(len(pop) * top)
    top_individuals = pop[0:top_index]
    lucky_indices = np.random.randint(top_index, len(pop), int(len(pop) * lucky))
    lucky_individuals =  [pop[i] for i in lucky_indices]
    top_individuals.extend(lucky_individuals)
    return top_individuals


# meta_data should be of the form :
# meta_data = { 'column_index' : {'node_name' : str() instance,
#                                'feature_importance': float() instance between 0 and 1,
#                                 'ancestor_graph': networkx.DiGraph() instance,
#                                },
#             }
def create_metadata(indices, node_names, feature_importance, ancestor_graphs):
    # All four inputs should be iterables.
    metadata = {}
    for index, node_name, feat_imp, ancestor_graph in zip(indices, node_names, feature_importance, ancestor_graphs):
        metadata[index] = {'node_name': node_name, 'feature_importance': feat_imp, 'ancestor_graph': ancestor_graph}
    return metadata


class Individual:
    __slots__ = ['data', 'fitness', 'meta_data']

    def __init__(self, data, meta_data, fitness=-1):
        self.data = data
        self.fitness = fitness
        self.meta_data = meta_data

    def extract_features(self):
        pass
    def merge(self, individual):
        pass

    def asexual_copy(self):
        individual = Individual(self.data, self.meta_data)
        return individual
