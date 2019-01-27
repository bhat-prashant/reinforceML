#
#
# # For now, it is assumed that input features are numerical !!
# # Future Work: add other meta information about each feature / chromosome
# def compute_meta_features(self, column):
#     if isinstance(column, np.ndarray):
#         self.value = column
#         self.mean = np.mean(column)
#         self.variance = np.var(column)
#     else:
#         raise Exception("Unknown data format. Expected \'numpy.ndarray\', Instead got {}", type(column))


# import networkx as nx
# import matplotlib.pyplot as plt
#
# e = [(1, 2), (5, 3), (6, 4), (2,7), (4,7), (3,7), (7, 8), (7, 10)]  # list of edges
# G = nx.DiGraph(e)
#
# G[1][2]  ['name'] = 'add'
# G[5][3]  ['name'] = 'sub'
# G[6][4]  ['name'] = 'mul'
# G[2][7]  ['name'] = 'div'
# G[4][7]  ['name'] = 'add'
# G[3][7]  ['name'] = 'log'
# G[7][8]  ['name'] = 'sqrt'
# G[7][10]  ['name'] = 'log'
#
# # nx.draw(G, with_labels=True)
# # plt.show()
#
#
# f = [(1, 2), (6, 4), (2,7), (4,7)]  # list of edges
# H = nx.DiGraph(f)
#
# H[1][2]  ['name'] = 'np. add'
# H[6][4]  ['name'] = 'np.mul'
# H[2][7]  ['name'] = 'np.div'
# H[4][7]  ['name'] = 'np.add'
#
# nx.draw(H, with_labels=True)
# plt.show()
#
#
# for x in G.nodes() :
#     y = G.predecessors(x)
#     print("")


#  Start from base.Fitness............................


# !/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"
