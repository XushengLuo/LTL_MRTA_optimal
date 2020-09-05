from collections import OrderedDict
import numpy as np
from datetime import datetime
from networkx.classes.digraph import DiGraph
from shapely.geometry import Point, LineString
from networkx.algorithms import dfs_labeled_edges
from numpy.random import uniform
from ShortestPath import shortest_path
import ws
import Buchi
from Visualization import path_plot
from sympy import symbols
import networkx as nx
from vis import vis


class tree(object):
    """ construction of prefix and suffix tree
    """

    def __init__(self, ts, buchi_graph, init, final, seg, env, accpet_list):
        """
        :param ts: transition system
        :param buchi_graph:  Buchi graph
        :param init: product initial state
        """
        # self.robot = 1
        self.goals = []
        self.ts = ts
        self.buchi_graph = buchi_graph
        self.init = init
        self.tree = DiGraph(type='PBA', init=init)
        label = self.label(init[0])
        self.tree.add_node(init, cost=0, label=label)

        self.group = dict()
        self.add_group(init)

        self.b_final = final
        self.finals_list = accpet_list
        self.seg = seg

        self.p = 0.9
        self.p_new = 0.9

        self.env = env
        self.connector_num = {'1': 1}
        self.connector = dict()

    def add_group(self, q_state):
        """
        group nodes with same buchi state
        :param q_state: new state added to the tree
        """
        try:
            self.group[q_state[1]].append(q_state)
        except KeyError:
            self.group[q_state[1]] = [q_state]

    def min2final(self, min_qb_dict, b_final, cand):
        """
         collects the buchi state in the tree with minimum distance to the final state
        :param min_qb_dict: dict
        :param b_final: feasible final state
        :return: list of buchi states in the tree with minimum distance to the final state
        """
        l_min = np.inf
        b_min = []
        for b_state in cand:
            if min_qb_dict[(b_state, b_final)] < l_min:
                l_min = min_qb_dict[(b_state, b_final)]
                b_min = [b_state]
            elif min_qb_dict[(b_state, b_final)] == l_min:
                b_min.append(b_state)
        return b_min

    def all2one(self, b_min):
        """
        partition nodes into 2 groups
        :param b_min: buchi states with minimum distance to the finals state
        :return: 2 groups
        """
        q_min2final = []
        q_minNot2final = []
        for b_state in self.group.keys():
            if b_state in b_min:
                q_min2final = q_min2final + self.group[b_state]
            else:
                q_minNot2final = q_minNot2final + self.group[b_state]
        return q_min2final, q_minNot2final

    def sample(self, buchi_graph, min_qb_dict):
        """
        sample point from the workspace
        :return: sampled point, tuple
        """

        # collects the buchi state in the tree with minimum distance to the final state
        b_min = self.min2final(min_qb_dict, self.b_final, self.group.keys())
        # partition of nodes
        q_min2final, q_minNot2final = self.all2one(b_min)
        # sample random nodes
        p_rand = np.random.uniform(0, 1, 1)
        q_rand = []
        if (p_rand <= self.p and len(q_min2final) > 0) or not q_minNot2final:
            q_rand = q_min2final[np.random.randint(0, len(q_min2final))]
        elif p_rand > self.p or not q_min2final:
            q_rand = q_minNot2final[np.random.randint(0, len(q_minNot2final))]
        # find feasible succssor of buchi state in q_rand
        Rb_q_rand = []
        label = self.tree.nodes[q_rand]['label']

        for b_state in buchi_graph.succ[q_rand[1]]:
            if self.checkTranB(q_rand[1], label, b_state):
                Rb_q_rand.append(b_state)
        # if empty
        if not Rb_q_rand:
            return Rb_q_rand, Rb_q_rand
        # collects the buchi state in the reachable set of qb_rand with minimum distance to the final state
        b_min = self.min2final(min_qb_dict, self.b_final, Rb_q_rand)

        # collects the buchi state in the reachable set of b_min with distance to the
        # final state equal to that of b_min - 1
        decr_dict = dict()
        for b_state in b_min:
            decr = []
            for succ in buchi_graph.succ[b_state]:
                if min_qb_dict[(b_state, self.b_final)] - 1 == min_qb_dict[(succ, self.b_final)] or succ == self.b_final:
                    decr.append(succ)
            decr_dict[b_state] = decr
        M_cand = [b_state for b_state in decr_dict.keys() if decr_dict[b_state]]
        # if empty
        if not M_cand:
            return M_cand, M_cand
        # sample b_min and b_decr
        b_min = M_cand[np.random.randint(0, len(M_cand))]
        b_decr = decr_dict[b_min][np.random.randint(0, len(decr_dict[b_min]))]
        truth = buchi_graph.edges[(b_min, b_decr)]['truth']
        x_rand = list(q_rand[0])
        suf_root = False
        if self.seg == 'suf' and self.init[1] in decr_dict[b_min]:
            suf_root = True
        # print(q_rand[1], b_min, b_decr, truth)
        return self.buchi_guided_sample_by_truthvalue(truth, x_rand, suf_root)

    def buchi_guided_sample_by_truthvalue(self, truth, x_rand, suf_root):
        """
        sample guided by truth value
        :param truth: the value making transition occur
        :param x_rand: random selected node
        :param x_label: label of x_rand
        :param regions: regions
        :return: new sampled point
        """
        # stay
        orig_rand = x_rand.copy()
        if truth == '1':
            for k in range(len(x_rand)):
                neig = list(self.env.neighbors(x_rand[k])) + [x_rand[k]]
                while True:
                    x_candidate = neig[np.random.randint(0, len(neig))]
                    if self.inner_collision(x_candidate, orig_rand, x_rand, k):
                        continue
                    x_rand[k] = x_candidate
                    break
            return tuple(x_rand)

        targets = {int(region_robot.split('_')[1]): region_robot.split('_')[0] for region_robot, t in truth.items() if t}
        for k in range(len(x_rand)):
            if k+1 not in targets.keys():
                if not suf_root or (suf_root and uniform(0, 1, 1) > self.p_new):
                    # if x_rand[k] not in x_rand[:k]:
                    #     continue
                    neig = list(self.env.neighbors(x_rand[k])) + [x_rand[k]]
                    while True:
                        x_candidate = neig[np.random.randint(0, len(neig))]
                        if self.inner_collision(x_candidate, orig_rand, x_rand, k):
                            continue
                        x_rand[k] = x_candidate
                        break
                else:
                    if x_rand[k] == self.init[0][k]:
                        continue
                    l, p = nx.single_source_dijkstra(self.env, source=x_rand[k], target=self.init[0][k])
                    target = p[1]
                    if target not in x_rand[:k]:
                        x_rand[k] = target
                    else:
                        neig = list(self.env.neighbors(x_rand[k])) + [x_rand[k]]
                        while True:
                            x_candidate = neig[np.random.randint(0, len(neig))]
                            if self.inner_collision(x_candidate, orig_rand, x_rand, k):
                                continue
                            x_rand[k] = x_candidate
                            break
            else:
                # select second point withi high probability
                if uniform(0, 1, 1) < self.p_new:
                    length = np.inf
                    path = []
                    # print(k, x_rand[k], truth)
                    # if x_rand[k] in ts['region'][targets[k+1]]:
                    #     continue
                    for cell in ts['region'][targets[k+1]]:
                        l, p = nx.single_source_dijkstra(self.env, source=x_rand[k], target=cell)
                        if l < length:
                            path = p
                            length = l
                    if len(path) > 1:
                        target = path[1]
                    if len(path) > 1 and target not in x_rand[:k]:
                        x_rand[k] = target
                    else:
                        neig = list(self.env.neighbors(x_rand[k])) + [x_rand[k]]
                        while True:
                            x_candidate = neig[np.random.randint(0, len(neig))]
                            if self.inner_collision(x_candidate, orig_rand, x_rand, k):
                                continue
                            x_rand[k] = x_candidate
                            break
                else:
                    neig = list(self.env.neighbors(x_rand[k])) + [x_rand[k]]
                    while True:
                        x_candidate = neig[np.random.randint(0, len(neig))]
                        if self.inner_collision(x_candidate, orig_rand, x_rand, k):
                            continue
                        x_rand[k] = x_candidate
                        break

        return tuple(x_rand)

    def inner_collision(self, x_candidate, orig_rand, x_rand, k):
        collision = False
        if x_candidate in x_rand[:k]:
            collision = True
            return collision
        else:
            for kk in range(k):
                if x_rand[kk] == orig_rand[k] and x_candidate == orig_rand[kk]:
                    collision = True
                    return collision
        return collision

    def extend(self, q_new, prec_list, label_new):
        """
        :param: q_new: new state form: tuple (mulp, buchi)
        :param: near_v: near state form: tuple (mulp, buchi)
        :param: obs_check: check obstacle free  form: dict { (mulp, mulp): True }
        :param: succ: list of successor of the root
        :return: extending the tree
        """

        added = 0
        cost = np.inf
        q_min = ()
        for pre in prec_list:
            dis = 0
            for k in range(len(q_new[0])):
                dis += 0 if q_new[0][k] == pre[0][k] else 1
            c = self.tree.nodes[pre]['cost'] + dis
            if c < cost:
                added = 1
                q_min = pre
                cost = c
        if added == 1:
            self.add_group(q_new)
            self.tree.add_node(q_new, cost=cost, label=label_new)
            self.tree.add_edge(q_min, q_new)
            if self.seg == 'pre':
                if q_new[1] in self.finals_list:
                    pred = list(self.tree.predecessors(q_new))[0]
                    node = (pred[0], q_new[1])
                    self.tree.add_node(node, cost=self.tree.nodes[pred]['cost'], label=self.tree.nodes[pred]['label'])
                    self.tree.add_edge(pred, node)
                    self.goals.append(node)
            else:
                if self.obs_check(q_new[0], self.init[0]) \
                        and self.checkTranB(q_new[1], self.tree.nodes[q_new]['label'], self.init[1]):
                    self.goals.append(q_new)

    def rewire(self, q_new, succ_list):
        """
        :param: q_new: new state form: tuple (mul, buchi)
        :param: near_v: near state form: tuple (mul, buchi)
        :param: obs_check: check obstacle free form: dict { (mulp, mulp): True }
        :return: rewiring the tree
        """
        for suc in succ_list:
            # root
            if suc == self.init:
                continue
            dis = 0
            for k in range(len(q_new[0])):
                dis += 0 if q_new[0][k] == suc[0][k] else 1
            c = self.tree.nodes[q_new]['cost'] + dis
            delta_c = self.tree.nodes[suc]['cost'] - c
            # update the cost of node in the subtree rooted at near_vertex
            if delta_c > 0:
                self.tree.remove_edge(list(self.tree.pred[suc].keys())[0], suc)
                self.tree.add_edge(q_new, suc)
                edges = dfs_labeled_edges(self.tree, source=suc)
                for u, v, d in edges:
                    if d == 'forward':
                        self.tree.nodes[v]['cost'] = self.tree.nodes[v]['cost'] - delta_c

    def prec(self, q_new):
        """
        find the predcessor of q_new
        :param q_new: new product state
        :return: label_new: label of new
        """
        p_prec = []
        for vertex in self.tree.nodes:
            if q_new != vertex and self.obs_check(vertex[0], q_new[0]) \
                    and self.checkTranB(vertex[1], self.tree.nodes[vertex]['label'], q_new[1]):
                p_prec.append(vertex)
        return p_prec

    def succ(self, q_new):
        """
        find the successor of q_new
        :param q_new: new product state
        :return: label_new: label of new
        """
        p_succ = []
        for vertex in self.tree.nodes:
            if q_new != vertex and self.obs_check(vertex[0], q_new[0]) \
                    and self.checkTranB(q_new[1], self.tree.nodes[q_new]['label'], vertex[1]):
                p_succ.append(vertex)
        return p_succ

    def obs_check(self, x0, x1):
        collision_free = True
        for k in range(len(x0)):
            if x0[k] != x1[k] and x0[k] not in self.env.neighbors(x1[k]):
                collision_free = False
                return collision_free
        return collision_free

    def label(self, x):
        """
        generating the label of position state
        :param x: position
        :return: label
        """
        label = []
        for i in range(len(x)):
            label.append('')
            # whether x lies within obstacle
            for obs, cells in self.ts['obs'].items():
                if x[i] in cells:
                    return ''

            # whether x lies within regions
            for region, cells in self.ts['region'].items():
                if x[i] in cells:
                    label[i] = region+'_{0}'.format(i+1)
                    break

        return label

    def checkTranB(self, b_state, x_label, q_b_new):
        """ decide valid transition, whether b_state --L(x)---> q_b_new
             :param b_state: buchi state
             :param x_label: label of x
             :param q_b_new buchi state
             :return True satisfied
        """

        b_state_succ = self.buchi_graph.succ[b_state]
        # q_b_new is not the successor of b_state
        if q_b_new not in b_state_succ:
            return False

        edge_label = self.buchi_graph.edges[(b_state, q_b_new)]['dnf']
        if edge_label.__str__() == '1':
            return True
        # without connector
        # table = {symbols(l): True for l in x_label if l != ''}
        # table.update({atom: False for atom in edge_label.atoms() if atom not in table.keys()})
        # if edge_label.subs(table):
        #     return True
        # else:
        #     return False

        # with connector
        table = dict()
        for l in x_label:
            if l != '':
                for atom in edge_label.atoms():
                    if l in atom.__str__():
                        table[atom] = True
        table.update({atom: False for atom in edge_label.atoms() if atom not in table.keys()})
        con_label = dict()
        if edge_label.subs(table):
            for atom in table:
                a = atom.__str__().split('_')
                if table[atom] and a[2] != '0':
                    if a[2] not in con_label.keys():
                        con_label[a[2]] = [a[1]]
                    else:
                        con_label[a[2]].append(a[1])
            for c, r in con_label.items():
                if c in self.connector.keys():
                    if len(set(r).intersection(self.connector[c])) < self.connector_num[c]:
                        return False
                else:
                    self.connector[c] = r
            return True
        else:
            return False

    def t_satisfy_b_truth(self, x_label, truth):
        """
        check whether transition enabled under current label
        :param x_label: current label
        :param truth: truth value making transition enabled
        :return: true or false
        """

        if truth == '1':
            return True

        true_label = [truelabel for truelabel in truth.keys() if truth[truelabel]]
        for label in true_label:
            if label not in x_label:
                return False

        false_label = [falselabel for falselabel in truth.keys() if not truth[falselabel]]
        for label in false_label:
            if label in x_label:
                return False

        return True

    def findpath(self, goals):
        """
        find the path backwards
        :param goals: goal state
        :return: dict path : cost
        """
        paths = OrderedDict()
        for i in range(len(goals)):
            goal = goals[i]
            path = [goal]
            s = goal
            while s != self.init:
                s = list(self.tree.pred[s].keys())[0]
                path.insert(0, s)
            # paths[i] = [self.tree.nodes[goal]['cost'], path]
            paths[i] = [self.path_cost(path), path]
            if self.seg == 'suf':
                dis = 0
                for r in range(len(path[-1][0])):
                    dis += 0 if path[-1][0][r] == self.init[0][r] else 1
                paths[i] = [self.path_cost(path) + dis, path + [self.init]]
        return paths

    def path_cost(self, path):
        """
        calculate cost
        :param path:
        :return:
        """
        cost = 0
        for k in range(len(path) - 1):
            dis = 0
            for r in range(len(path[k + 1][0])):
                dis += 0 if path[k + 1][0][r] == path[k][0][r] else 1
            cost += dis
        return cost


def construction_tree(bias_tree, buchi_graph, min_qb_dict):
    while bias_tree.tree.number_of_nodes() < 10000:

        x_new = bias_tree.sample(buchi_graph, min_qb_dict)
        if not x_new[0]:
            continue
        # sample
        label_new = bias_tree.label(x_new)

        for b_state in buchi_graph.nodes():

            q_new = (x_new, b_state)
            if q_new not in bias_tree.tree.nodes():
                # candidate parent state
                prec = bias_tree.prec(q_new)
                bias_tree.extend(q_new, prec, label_new)
            # rewire
            if q_new in bias_tree.tree.nodes():
                # only rewire within the subtree, it may affect the node which is a root
                succ = bias_tree.succ(q_new)
                bias_tree.rewire(q_new, succ)

            if len(bias_tree.goals) > 0:
                cost_pat = bias_tree.findpath(bias_tree.goals)
                return cost_pat


def bias_sampling_alg(buchi_graph, init, ts, min_qb, env):
    """
    build multiple subtree by transferring
    :param todo: new subtask built from the scratch
    :param buchi_graph:
    :param init: root for the whole formula
    :param todo_succ: the successor of new subtask
    :param ts:
    :param no:
    :param centers:
    :param max_node: maximum number of nodes of all subtrees
    :param subtask2path: (init, end) --> init, p, ..., p, end
    :param starting2waypoint: init --> (init, end)
    :param newsubtask2subtask_p: new subtask need to be planned --> new subtask noneed
    :return:
    """
    start = datetime.now()
    print('--------------finding path for the prefix part ---------------------')
    bias_tree = tree(ts, buchi_graph, init,
                 buchi_graph.graph['accept'][np.random.randint(0, len(buchi_graph.graph['accept']))], 'pre', env,
                 buchi_graph.graph['accept'])
    cost_path_pre = construction_tree(bias_tree, buchi_graph, min_qb)
    pre_time = (datetime.now() - start).total_seconds()
    start = datetime.now()
    opt_cost = (np.inf, np.inf)
    opt_path_pre = []
    opt_path_suf = []

    for i in range(1):
        # goal product state
        goal = bias_tree.goals[i]
        tree_suf = tree(ts, buchi_graph, goal, goal[1], 'suf', env, [])

        label_new = bias_tree.label(goal)

        if tree_suf.checkTranB(tree_suf.init[1], label_new, tree_suf.init[1]):
            opt_path_pre = cost_path_pre[i][1]  # plan of [(position, buchi)]
            path_pre = {type_robot: [] for type_robot in workspace.num2team.values()}
            for k in range(len(opt_path_pre)):
                for num, type_robot in workspace.num2team.items():
                    path_pre[type_robot].append(opt_path_pre[k][0][num - 1])

            path_suf = {type_robot: [] for type_robot in workspace.num2team.values()}
            return pre_time, cost_path_pre[i][0], path_pre, path_suf

        # update accepting buchi state
        # buchi_graph.graph['accept'] = goal[1]
        # construct suffix tree
        print('--------------suffix path for {0}-th goal (of {1} in total)---------------------'.
        format(i, len(bias_tree.goals)))
        cost_path_suf_cand = construction_tree(tree_suf, buchi_graph, min_qb)

        print('{0}-th goal: {1} accepting goals found'.format(i, len(tree_suf.goals)))
        # couldn't find the path
        try:
            # order according to cost
            cost_path_suf_cand = OrderedDict(sorted(cost_path_suf_cand.items(), key=lambda x: x[1][0]))
            mincost = list(cost_path_suf_cand.keys())[0]
        except IndexError:
            del cost_path_pre[i]
            print('delete {0}-th item in cost_path_pre, {1} left'.format(i, len(cost_path_pre)))
            continue
        cost_path_suf = cost_path_suf_cand[mincost]

        if cost_path_pre[i][0] + cost_path_suf[0] < opt_cost[0] + opt_cost[1]:
            opt_path_pre = cost_path_pre[i][1]  # plan of [(position, buchi)]
            opt_path_suf = cost_path_suf[1]
            opt_cost = cost_path_pre[i][0] + cost_path_suf[0]  # optimal cost (pre_cost, suf_cost)

        suf_time = (datetime.now() - start).total_seconds()

        path_pre = {type_robot: [] for type_robot in workspace.num2team.values()}
        for k in range(len(opt_path_pre)):
            for num, type_robot in workspace.num2team.items():
                path_pre[type_robot].append(opt_path_pre[k][0][num - 1])

        path_suf = {type_robot: [] for type_robot in workspace.num2team.values()}
        for k in range(len(opt_path_suf)):
            for num, type_robot in workspace.num2team.items():
                path_suf[type_robot].append(opt_path_suf[k][0][num - 1])

        return pre_time + suf_time, opt_cost, path_pre, path_suf


workspace = ws.Workspace()
ts = {'region': workspace.regions, 'obs': workspace.obstacles}

buchi = Buchi.buchi_graph(workspace.formula, workspace.formula_comp, workspace.contra)
buchi.formulaParser()
buchi.execLtl2ba()
_ = buchi.buchiGraph()
buchi.DelInfesEdge()
min_qb = buchi.MinLen()
buchi.FeasAcpt(min_qb)
buchi_graph = buchi.buchi_graph

time = []
cost = []
opt_path = []
opt_cost = np.inf
for i in range(1):
    tm, c, robot_path_pre, robot_path_suf = bias_sampling_alg(buchi_graph, (workspace.init_state, buchi_graph.graph['init'][0]), ts,
                                 min_qb, workspace.graph_workspace)

    robot_path = {robot: path + robot_path_suf[robot][1:] + robot_path_suf[robot][1:] for
                  robot, path in robot_path_pre.items()}
    print('---------------- plotting -----------------------')
    vis(workspace, robot_path, {robot: [len(path)] * 2 for robot, path in robot_path.items()}, [])

    # path_plot(opt_path, ts['region'], ts['obs'])

#     # print(tm, c)
#     time.append(tm)
#     cost.append(c)
#     if c < opt_cost:
#         opt_cost = c
#         opt_path = p
# # for p in opt_path:
# #     print(t.label(p[0]), ' ', end='')
# print(np.mean(time), )
# # path_plot(opt_path, regions, obs, num_grid)
