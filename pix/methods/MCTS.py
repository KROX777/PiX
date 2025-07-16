"""
A minimal implementation of Monte Carlo tree search (MCTS) in Python 3
Luke Harold Miles, July 2019, Public Domain Dedication
Original Codes: https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1

Modified by Mingquan Feng, Sept 2023.
Modification: 1) remove all inverted reward.

Modified by Chuyang Xiang, Jul 2025.
Modification: 1) adaptive change for the new calculator and hypotheses tree architecture.
"""

import traceback
import sys
import warnings
from collections import defaultdict
import math
import random
from joblib import Parallel, delayed
import numpy as np
import os
import copy
import logging

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from pix.hypotheses_tree import LightTree
from pix.methods.BFSearch import single_test
from ordered_set import OrderedSet
from pix.utils.others import save_load, timing, timing_with_return

class SearchNode():
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """
    def __init__(self, cfg, root_dir, deci_list=None):
        self.deci_list = deci_list
        self.reward_ = None
        self.children = None
        self.cfg = cfg
        self.root_dir = root_dir
        self.hypo_tree = LightTree(cfg, root_dir)
        if self.deci_list is None:
            self.deci_list = [0]
            for id in self.hypo_tree.root.children_nodes:
                self.deci_list.append(id)

    def find_children(self):
        """
        All possible successors of this deci state
        Return a set of children, the set is ordered for reproducibility.
        """
        child_list = []
        for deci in self.deci_list:
            flag = True
            for child in self.hypo_tree.nodes[deci].children_nodes:
                if child in self.deci_list:
                    flag = False
                    break
            if flag == False:
                continue 
            for child in self.hypo_tree.nodes[deci].children_nodes:
                new_deci_list = copy.deepcopy(self.deci_list)
                new_deci_list.append(child)
                child_list.append(SearchNode(cfg=self.cfg, root_dir=self.root_dir, deci_list=new_deci_list))
                
        if len(child_list) > 0:
            return OrderedSet(child_list)
        else:
            return None

    def find_random_child(self):
        "Random successor of this deci state (for more efficient simulation)"
        if self.is_terminal():
            return None
        else:
            return random.choice(list(self.find_children()))

    def reward(self):
        "Assumes `self` is terminal node, return negative MSE as reward."
        if not self.is_terminal():
            raise RuntimeError(f"reward called on nonterminal self {self}")
        else:
            if self.reward_ is None:
                mse = single_test(cfg=self.cfg, root_dir=self.root_dir, deci_list=self.deci_list)['train_loss']
                self.reward_ = 1 - self._normalize_mse_log(mse)
                if np.isnan(self.reward_):
                    self.reward_ = -1e-5
            return self.reward_

    def _normalize_mse_log(self, mse, min_mse=1e-4, max_mse=1):
        # Apply logarithmic scaling
        mse = np.clip(mse, min_mse, max_mse)
        log_mse = np.log(mse)
        log_min = np.log(min_mse)
        log_max = np.log(max_mse)
        return (log_mse - log_min) / (log_max - log_min)

    def is_terminal(self):
        "Returns True if the node has no children"
        return self.find_children() is None

    def __str__(self) -> str:
        return str(self.deci_list)

    def __hash__(self):
        "Nodes must be hashable"
        return hash(str(self))

    def __eq__(node1, node2):
        "Nodes must be comparable"
        return str(node1) == str(node2)

class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self, cfg, root_dir, exploration_weight=1, epsilon=0.05, max_rollout=None, n_jobs=5, out_file=None, verbose_level=1):
        self.cfg = cfg
        self.root_dir = root_dir
        self.exploration_weight = exploration_weight
        self.max_rollout = max_rollout
        self.MAX_DEPTH = 100
        self.epsilon = epsilon
        self.n_jobs = n_jobs
        self.out_file = out_file
        self.datafold_tuple=(0,1) #int tuple, (k-th fold, num of total folds), used for k-fold CV
        self.verbose_level = verbose_level
        if self.max_rollout is None:
            self.max_rollout = 60
        
        # 设置日志器
        self.logger = logging.getLogger(__name__)

    def init_search(self):
        self.Q_list = defaultdict(list)  # all rewards of each node
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.reward_cache = dict() #reward of each terminal node

    @timing_with_return
    def search(self, init_node=None, do_initialize=True,seed=0):
        if self.verbose_level > 0:
            self.logger.info(f"Searching: {self.epsilon=}, {self.max_rollout=}, {self.n_jobs=}, {seed=}")
            self.logger.info("--- ---")

        random.seed(seed)
        if init_node is None:
            init_node = SearchNode(cfg=self.cfg, root_dir=self.root_dir)
        node = init_node
        if do_initialize:
            self.init_search()

        for i in range(self.MAX_DEPTH):
            children = node.find_children()
            if children is None:
                break
            if len(children)==1: 
                # single child
                node = list(children)[0]
            else:  
                # multiple children
                #TODO: n_rollout should be less than num of all leafs.
                n_rollout = min(self.max_rollout, len(node.hypo_tree.gen_all_paths_from_deci_list(node.deci_list)))
                if self.n_jobs <= 1:  #single-process
                    for _ in range(n_rollout):
                        self._do_rollout(node)
                else: # multi-process
                    self._para_do_rollout(node, n_rollout)
                node = self._choose(node)

            if self.verbose_level>0:
                self.logger.debug(f"Step {i}")
                self.logger.debug(f"Decision list: {node.deci_list}")
                self.logger.debug("--- ---")

        if self.verbose_level > 0:
            self.logger.info("Search result node:")
            sol = single_test(cfg=self.cfg, root_dir=self.root_dir, deci_list=node.deci_list)
            self.logger.info(f"Search result: {sol}")

        self.checkpoint = f"./model/MCTS/max_rollout={self.max_rollout},seed={seed}.pkl"
        save_load(self.checkpoint, self, is_save=True)

        return node

    def get_top_k(self, k=3):
        """ returns a dict(node, sol) of top_k reward nodes.
        """
        self.logger.info(f"Number of tested nodes: {len(self.reward_cache)}")  # num of testing?
        top_k_nodes = sorted(self.reward_cache.items(), key=lambda x: x[1], reverse=True)[:k]
        top_k_sols = dict()
        if self.verbose_level>0:
            for i in range(k):
                node = top_k_nodes[i][0]
                self.logger.info(f"-----Top {i+1} node, reward={self.reward_cache[node]}-----")
                sol = single_test(self.cfg, self.root_dir, node.deci_list)
                self.logger.info(f"Top {i+1} solution: {sol}")
                top_k_sols[node] = sol
        return top_k_sols

    @timing
    def k_fold_cv_search(self, k=5, seed=0):
        '''
        k_fold cross validation MCTS.
        return a list of dict(node, sol)
        '''
        # self.logger.debug(f"start k_fold_cv, {n_jobs * self.n_jobs} cpu cores needed")
        old_datafold_tuple = self.datafold_tuple

        def one_fold_search(i):
            self.logger.info(f"=== {i+1}-th fold ===")
            self.datafold_tuple = (i, k) #BUG: cannot modify class vars in sub-process.
            _, search_time = self.search(do_initialize=True, seed=seed)
            return self.get_top_k(), search_time

        top_k_sols_list, search_time_list = zip(*[one_fold_search(i) for i in range(k)])

        self.datafold_tuple = old_datafold_tuple

        #--summary and print
        self.logger.info("----- k_fold_cv_search results -----")
        results_list, time_mean_std = self.summary(top_k_sols_list, search_time_list)
        for r in results_list:
            self.logger.info(f"CV result: {r}")
        self.logger.info("Search time mean={}, std={}".format(*time_mean_std))

        return results_list, time_mean_std

    def summary(self, top_k_sols_list, search_time_list):
        '''
        Input:
            top_k_sols_list, a list of dict(node, sol)
            search_time_list, a list of float
        Output:
            results_list, list of result_dict, sorted by valid_loss_mean
                result_dict = {
                    hit_ratio: (occurence in all folds) / n_folds,  occurence<2 is filtered.
                    train_loss_mean_std: float tuple.
                    valid_loss_mean_std: float tuple.
                    deci_list: dict }
            time_mean_std, float tuple.
        '''
        mean_std_func = lambda lst: (np.mean(lst), np.std(lst,ddof=1))
        time_mean_std = mean_std_func(search_time_list)

        tmp_results_dict = defaultdict(lambda: {"train_loss_list":[], "valid_loss_list":[]})
        for top_k_sols in top_k_sols_list:
            for node, sol in top_k_sols.items():
                tmp_results_dict[node]["train_loss_list"].append(sol["train_loss"])
                tmp_results_dict[node]["valid_loss_list"].append(sol["valid_loss"])

        results_list = []
        n_folds = len(top_k_sols_list)
        for node, result in tmp_results_dict.items():
            train_loss = result["train_loss_list"]
            valid_loss = result["valid_loss_list"]
            occurence = len(train_loss)
            if occurence < 2:
                continue  #item with occurence<2 is ignored.
            result_dict = {
                "train_loss_mean_std": mean_std_func(train_loss),
                "valid_loss_mean_std": mean_std_func(valid_loss),
                "hit_ratio": occurence/n_folds,
                "deci_list": node.deci_list
            }
            results_list.append(result_dict)
        results_list.sort(key=lambda x:x["valid_loss_mean_std"][0])

        return results_list, time_mean_std

    @timing
    def _para_do_rollout(self, node, n_rollout):
        "Make the tree one layer better. (parallelized n_jobs)"
        n_batch = math.ceil(n_rollout / self.n_jobs)
        for _ in range(n_batch):
            path_batch = []
            for __ in range(self.n_jobs):
                path = self._select(node)
                path_batch.append(path)
                leaf = path[-1]
                self._expand(leaf)

            reward_batch = Parallel(n_jobs=self.n_jobs)(delayed(self._simulate)(path[-1]) for path in path_batch)

            # update tree infos.
            for i, path in enumerate(path_batch):
                reward, terminal_node = reward_batch[i]
                self._backpropagate(path, reward)
                self.reward_cache[terminal_node] = reward

    def _value(self, node):
        if self.N[node] == 0:
            return float("-inf")  # avoid unseen moves
        #return self.Q[node] / self.N[node]  # average reward
        # top quantile Q reward

        sorted_Q_list = sorted(self.Q_list[node],reverse=True)
        top_quantile = int(max(1, len(sorted_Q_list)* self.epsilon))
        sorted_Q_list = sorted_Q_list[:top_quantile]
        return np.mean(sorted_Q_list)

    def _choose(self, node):
        "Choose the best successor of node. (Choose a move in the game, Action for 1 iteration)"
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children:
            return node.find_random_child()

        children = self.children[node]
        if self.verbose_level>1:
            self.logger.debug(f"choose of {str(node)=} ---")
            self.logger.debug("children=" + str([str(n) for n in children]))
            self.logger.debug("Q[children]=" + str([self.Q[n] for n in children]))
            self.logger.debug("Q_list[children]=" + str([self.Q_list[n] for n in children]))
            self.logger.debug("N[children]=" + str([self.N[n] for n in children]))
            self.logger.debug("Q/N[children]=" + str([self.Q[n]/max(self.N[n],1) for n in children])) # avoid division by zero
            self.logger.debug("risk-seek-obj=" + str([self._value(n) for n in children]))


        return max(children, key=self._value)  # the node with the highest quantile score

    def _do_rollout(self, node):
        "Make the tree one layer better. (Train for one iteration.)"
        path = self._select(node)

        leaf = path[-1]
        self._expand(leaf)

        reward, terminal_node = self._simulate(leaf)

        self.reward_cache[terminal_node] = reward
        self._backpropagate(path, reward)

    def _select(self, node):
        """Find an unexplored descendent of `node`
        Return a path, with last node being either unexplored or terminal.
        """
        path = []
        while True:
            if self.n_jobs > 1: #parallel case, virtual loss, ref: https://www.moderndescartes.com/essays/deep_dive_mcts/#virtual-losses
                self.N[node] += 1  #add N to avoid repeated selection.
                self.Q[node] -= 1  #virtual loss, minus Q to avoid repeated seletion.(will be added back later in _backpropagate())
            path.append(node)

            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
                return path

            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                node = unexplored.pop() # descend a layer deeper, to an unexplored child
            else:
                node = self._uct_select(node)  # descend a layer deeper, via UCT among all children.

    def _expand(self, node):
        """Update the `children` dict with the children of `node`
        Mark the node as explored.
        """
        if node in self.children:
            return  # already expanded
        self.children[node] = node.find_children()

    def _simulate(self, node):
        "Returns the reward and terminal_node for a random simulation (to completion) of `node`"
        while True:
            if node.is_terminal():
                if node in self.reward_cache:
                    reward = self.reward_cache[node]
                else:
                    reward = node.reward()
                break
            else:
                node = node.find_random_child()
        terminal_node = node
        return reward, terminal_node

    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            if self.n_jobs > 1: # parallel case, virtual loss, ref: https://www.moderndescartes.com/essays/deep_dive_mcts/#virtual-losses
                self.Q[node] += reward + 1  #add back Previous virtual loss.
                self.Q_list[node].append(reward)
            else:  #single-process case
                self.N[node] += 1
                self.Q[node] += reward
                self.Q_list[node].append(reward)  # 添加这一行！

    def _uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"

        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])

        log_N_vertex = math.log(self.N[node])

        def uct(n):
            "Upper confidence bound for trees"
            return self.Q[n]/self.N[n] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )

        return max(self.children[node], key=uct)
