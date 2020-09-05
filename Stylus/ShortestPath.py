from __future__ import division

# from Constree import tree
from networkx.classes.digraph import DiGraph
import networkx as nx
import numpy as np
# from Problem import problemFormulation
# from Visualization import path_plot
import math


# def direction(c, r, tree):
#     feas_point = []
#     # left
#     if (c[0]-r)>0 and 'o' not in tree.label((c[0]-r, c[1])):
#         feas_point.append((round(c[0]-r, 10), round(c[1], 10)))
#     # right
#     if (c[0]+r)<1 and 'o' not in tree.label((c[0]+r, c[1])):
#         feas_point.append((round(c[0]+r,10), round(c[1],10)))
#     # up
#     if (c[1]+r)<1 and 'o' not in tree.label((c[0], c[1]+r)):
#         feas_point.append((round(c[0],10), round(c[1]+r,10)))
#     # down
#     if (c[1] - r) >0 and 'o' not in tree.label((c[0], c[1] - r)):
#         feas_point.append((round(c[0],10), round(c[1] - r, 10)))
#
#     return feas_point


def shortest_path(env, s, t):
    path = nx.shortest_path(env, source=s, target=t)

    target = path[-1]
    for k in range(len(path)-2):
        p1 = np.array(path[k])
        p2 = np.array(path[k+1])
        p3 = np.array(path[k+2])
        if np.inner(p2-p1, p3-p1)/np.linalg.norm(p2-p1)/np.linalg.norm(p3-p1) < 0.95:
            target = tuple(path[k+1])
            break
    return target

    # path_fake = []
    # for p in path:
    #     path_fake.append((p, '0'))
    # # print(s, t, target)
    # # path_plot(path_fake, regions, obs, num_grid)
    # return target
