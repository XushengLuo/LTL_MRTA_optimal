"""
__author__ = chrislaw
__project__ = TransferLTL
__date__ = 2/1/19
"""

from shapely.geometry import Point, Polygon
import sys
import random
import numpy
import math
import itertools
import networkx as nx


class Workspace(object):
    def __init__(self):

        self.type_num = {1: 3, 2: 2}   # single-task robot
        self.team = dict()
        num = 0
        for t, r in self.type_num.items():
            self.team[t] = list(range(num+1, num+r+1))
            num += r

        self.num2team = {nums[i]: (t, i) for t, nums in self.team.items() for i in range(len(nums))}
        self.width = 9
        self.length = 9
        self.regions = {'l{0}'.format(i + 1): j for i, j in enumerate(self.allocate_region_dars())}
        self.obstacles = {'o{0}'.format(i + 1): j for i, j in enumerate(self.allocate_obstacle_dars())}
        self.graph_workspace = nx.Graph()
        self.build_graph()

        self.type_robot_location = self.initialize()
        # self.type_robot_location[(1, 0)] = (0, 0)
        # self.type_robot_location[(1, 1)] = (1, 0)
        # self.type_robot_location[(1, 2)] = (2, 0)

        self.init_state = tuple([self.type_robot_location[self.num2team[i]] for i in range(1, len(self.num2team)+1)])

        # self.formula = '<> e1 && <> (e2  && <> (e3 && <> ( e4 && <> e5)))'
        # self.formula = '<> (e1 && <> e2)'
        # self.groups = {1: 'l2_1_1_1',
        #                2: 'l3_1_1_1'}
        self.formula = '<> ((e1 && ! e2) && <> e2) && <> e3 && (! e2 U e3)'
        self.groups = {1: 'l2_1_2_1',
                       2: 'l3_1_2_1',
                       3: 'l4_2_1_0'}
        self.formula_comp = {}
        for index, group in self.groups.items():
            region_type_num = group.split('_')
            comb = list(itertools.combinations(self.team[int(region_type_num[1])], int(region_type_num[2])))
            case = '(' + ' || '.join('(' + ' && '.join([region_type_num[0] + '_' + str(r) + '_' + region_type_num[3]
                                                        for r in comb[i]]) + ')'
                               for i in range(len(comb))) + ')'
            self.formula_comp[index] = case
        print(self.formula_comp)
        self.contra = [('e1', 'e2')]

    def allocate_region_dars(self):
        # (x, y) --> (y-1, 30-x)
        # ijrr
        # # small workspace
        regions = []
        regions.append(list(itertools.product(range(6, 9), range(0, 2))))  # l1
        regions.append(list(itertools.product(range(7, 9), range(5, 8))) + [(8,4)])  # l2
        regions.append(list(itertools.product(range(0, 3), range(0, 4))))  # l3
        regions.append(list(itertools.product(range(0, 4), range(6, 7))))  # l4
        regions.append(list(itertools.product(range(4, 6), range(8, 9))))  # l5

        return regions

    def allocate_obstacle_dars(self):

        # small workspace
        obstacles = []
        # obstacles.append(list(itertools.product(range(3, 4), range(2, 6))) + [(4, 2)])  # o1
        obstacles.append(list(itertools.product(range(4, 5), range(0, 6))))  # o1

        return obstacles

    def reachable(self, location, obstacles):
        next_location = []
        # left
        if location[0]-1 > 0 and (location[0]-1, location[1]) not in obstacles:
            next_location.append((location, (location[0]-1, location[1])))
        # right
        if location[0]+1 < self.width and (location[0]+1, location[1]) not in obstacles:
            next_location.append((location, (location[0]+1, location[1])))
        # up
        if location[1]+1 < self.length and (location[0], location[1]+1) not in obstacles:
            next_location.append((location, (location[0], location[1]+1)))
        # down
        if location[1]-1 > 0 and (location[0], location[1]-1) not in obstacles:
            next_location.append((location, (location[0], location[1]-1)))
        return next_location

    def build_graph(self):
        obstacles = list(itertools.chain(*self.obstacles.values()))
        for i in range(self.width):
            for j in range(self.length):
                if (i, j) not in obstacles:
                    self.graph_workspace.add_edges_from(self.reachable((i, j), obstacles))

    # def initialize(self):
    #     type_robot_location = {(1, 0): (7, 0), (1, 1): (7, 1), (1, 2): (8, 1),
    #                            (2, 0): (6, 0), (2, 1): (6, 1)}
    #     return type_robot_location

    def initialize(self):
        type_robot_location = dict()
        # x0 = list(itertools.product(range(5, 9), range(9)))
        x0 = [(i, j) for i in range(9) for j in range(9) for k in range(1, 6)
              if (i, j) not in self.obstacles['o1'] and (i, j) not in self.regions['l'+str(k)]]
        for robot_type in self.type_num.keys():
            for num in range(self.type_num[robot_type]):
                while True:
                    candidate = random.sample(x0, 1)[0]
                    if candidate not in type_robot_location.values() and candidate not in self.regions['l2']\
                            and candidate not in self.regions['l5']:
                        type_robot_location[(robot_type, num)] = candidate
                        x0.remove(candidate)
                        break
        return type_robot_location

