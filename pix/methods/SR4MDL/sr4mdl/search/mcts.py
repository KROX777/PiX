import json
import time
import torch
import random
import logging
import sklearn
import traceback
import numpy as np
import nd2py as nd
import pandas as pd
from tqdm import tqdm
from typing import List, Generator, Tuple, Dict
import re
import sys
import os
import copy

# Add the root directory to sys.path to make pix module importable
_current_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.join(_current_dir, '../../../../..')
_root_dir = os.path.abspath(_root_dir)
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from ..env import simplify
# from ..env.symbols import *
from ..env.tokenizer import Tokenizer
from ..model.mdlformer import MDLformer
from .utils import preprocess, sample_Xy
from nd2py.utils import seed_all, Timer, NamedTimer, R2_score, RMSE_score
from pix.utils.scipy_utils import optimize_with_timeout

def calculate_nesting_depth(expr):
    if expr is None:
        return 0
    if isinstance(expr, (nd.Variable, nd.Number)):
        return 1
    if hasattr(expr, 'operands') and expr.operands:
        max_depth = max(calculate_nesting_depth(operand) for operand in expr.operands)
        return max_depth + 1
    return 1

class Node:
    def __init__(self, eqtrees:List[nd.Symbol]):
        # Formula part
        self.eqtrees = eqtrees
        self.phi = None
        self.complexity = None
        self.reward = None
        self.r2 = None
        self.mse = None

        # MC Tree part
        self.parent = None
        self.children = []
        self.N = 0
        self.Q = 0
        self.MDL = None
    
    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return '[' + ', '.join(str(eq) for eq in self.eqtrees) + ']' + f' (N={self.N}, Q={self.Q/(self.N+1e-6):.2f}, MDL={self.MDL:.0f})'
    
    def to_route(self, N=5, c=1.41) -> str:
        """
        Root
        ├ Node1
        ┆ ├ self
        ┆ └ Node1-2
        └ Node2
        """
        rev_route = [self]
        tmp = self
        while tmp.parent:
            rev_route.append(tmp.parent)
            tmp = tmp.parent
        items = []
        for node in rev_route:
            if node.parent:
                siblings = node.parent.children
                UCT = {x: x.Q/(x.N+1e-6) + c / max(x.MDL, 1e-6) * np.sqrt(node.N) / (x.N+1) for x in siblings}
                siblings = sorted(siblings, key=UCT.get, reverse=True)
                siblings = siblings[:N]
            else:
                siblings = [node]
                UCT = {node: 0.0}
            new_items = [f'{node} (MDL={node.MDL:.2f}, UCT={UCT[node]:.2f})' for node in siblings]
            self_idx = siblings.index(node)
            for idx, item in enumerate(items): 
                items[idx] = ('├ ' if idx < len(items)-1 else '└ ') + item.replace('\n', '\n' + ('┆ ' if idx < len(items)-1 else '  '))
            new_items[self_idx] = '\033[31m' + new_items[self_idx] + '\033[0m' + ('\n' if items else '') + '\n'.join(items)
            items = new_items
        assert len(items) == 1
        return items[0]

    def copy(self) -> 'Node':
        copy = Node([eqtree.copy() for eqtree in self.eqtrees])
        copy.phi = self.phi
        copy.complexity = self.complexity
        copy.reward = self.reward
        copy.r2 = self.r2
        copy.mse = self.mse
        copy.MDL = self.MDL
        return copy
        
class MCTS4MDL(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    """
    Monte Carlo Tree Search from the sample side, only use f but not g
    """
    def __init__(self,
                 tokenizer:Tokenizer,
                 model:MDLformer,
                 binary:List[str|nd.Symbol]=[nd.Add, nd.Sub, nd.Mul, nd.Div, nd.Max, nd.Min],
                 unary:List[str|nd.Symbol]=[nd.Sqrt, nd.Log, nd.Abs, nd.Neg, nd.Inv, nd.Sin, nd.Cos, nd.Tan],
                 leaf:List[float|nd.Number]=[nd.Number(1), nd.Number(0.5)],
                 const_range=None,
                 child_num=50,
                 n_playout=100,
                 d_playout=10,
                 max_len=30,
                 c=1.41,
                 n_iter=100,
                 sample_num=300,
                 log_per_iter:float=float('inf'),
                 log_per_sec:float=float('inf'),
                 save_path:str|None=None,
                 keep_vars:bool=False,
                 normalize_y:bool=False,
                 normalize_all:bool=False,
                 remove_abnormal:bool=False,
                 random_state:int=42,
                 train_eval_split:float=1.0,
                 eta:float=0.999,
                 disable_prod_model:bool=False,
                 max_power_abs:float|None=None,
                 complexity_limit:int|None=None,
                 complexity_alpha:float=1.0,
                 max_nesting_depth:int|None=None,
                 calculator=None,
                 x_name=None,
                 y_name=None,
                 other_params_name=None,
                 deci_list_len=None,
                 mode='inverse',
                 log_every_n_iters:int|None=None,
                 **kwargs):
        self.tokenizer = tokenizer # Tokenizer(-100, 100, 4)
        self.model = model # DataEncoder(AttrDict({}), self.tokenizer.get_token_list(all=False))
        self.max_var = self.model.args.max_var

        self.eqtree = None
        self.binary = [eval(x, globals(), nd.__dict__) if isinstance(x, str) else x for x in binary]
        self.unary = [eval(x, globals(), nd.__dict__) if isinstance(x, str) else x for x in unary]
        self.leaf = [nd.Number(x) if isinstance(x, float) else x for x in leaf]
        self.variables = []
        
        # Debug: Print actual operators being used
        print(f"[MCTS DEBUG] Binary operators: {[str(x) for x in self.binary]}")
        print(f"[MCTS DEBUG] Unary operators: {[str(x) for x in self.unary]}")
        print(f"[MCTS DEBUG] Leaf constants: {[str(x) for x in self.leaf]}")

        self.const_range = const_range
        self.child_num = child_num
        self.n_playout = n_playout
        self.d_playout = d_playout
        self.max_len = max_len
        self.c = c
        self.n_iter = n_iter
        self.sample_num = sample_num

        self.log_per_iter = log_per_iter
        self.log_per_sec = log_per_sec
        self.records = []
        self.logger = logging.getLogger(__name__)
        self.step_timer = Timer()
        self.view_timer = Timer()
        self.named_timer = NamedTimer()
        self.save_path = save_path
        self.keep_vars = keep_vars
        self.normalize_y = normalize_y
        self.normalize_all = normalize_all
        self.remove_abnormal = remove_abnormal
        self.random_state = random_state
        self.train_eval_split = train_eval_split
        self.eta = eta
        self.disable_prod_model = disable_prod_model
        self.max_power_abs = max_power_abs
        self.complexity_limit = complexity_limit
        self.complexity_alpha = complexity_alpha
        self.max_nesting_depth = max_nesting_depth
        
        self.calculator = calculator
        self.y_name = y_name
        self.x_name = x_name
        self.other_params_name = other_params_name
        self.deci_list_len = deci_list_len
        self.mode = mode
        self.log_every_n_iters = log_every_n_iters
        
        self.optimization_cache = {}
        self._global_top = {}
        self.physics_eval_cache = {}
        
        if kwargs:
            self.logger.warning('Unknown args: %s', ', '.join(f'{k}={v}' for k,v in kwargs.items()))

    def _compute_expression_hash(self, node: Node) -> str:
        import hashlib
        expr_strs = []
        for eq in node.eqtrees:
            if hasattr(eq, "to_str"):
                expr_str = eq.to_str()
            else:
                expr_str = str(eq)
            expr_strs.append(expr_str)
        
        expr_strs.sort()
        combined_str = "|".join(expr_strs)

        return hashlib.md5(combined_str.encode()).hexdigest()

    # ---------- Global Top Helpers ----------
    def _register_global_candidate(self, node: 'Node'):
        try:
            if node is None or node.reward is None or not np.isfinite(node.reward):
                return
            h = self._compute_expression_hash(node)
            if h not in self._global_top or (self._global_top[h].reward or -np.inf) < node.reward:
                self._global_top[h] = node.copy()
        except Exception:
            pass

    def _get_global_top_k_nodes(self, k:int=10):
        nodes = list(self._global_top.values())
        if not nodes:
            return []
        nodes.sort(key=lambda n: n.reward if n.reward is not None else -np.inf, reverse=True)
        return nodes[:k]

    def get_cache_stats(self) -> dict:
        return {
            'cache_size': len(self.optimization_cache),
            'cached_expressions': list(self.optimization_cache.keys())
        }
    
    def clear_cache(self):
        self.optimization_cache.clear()

    def __repr__(self):
        res = 'None' if self.eqtree is None else self.eqtree.to_str()
        return '{}({})'.format(self.__class__.__name__, res)

    def fit(self, X:np.ndarray|pd.DataFrame|Dict[str,np.ndarray], y, n_iter=None, use_tqdm=False, 
            early_stop:callable=lambda r2, complexity, eq: r2 > 0.99999, is_final_optim=True, max_iter=2):
        """
        Args:
            X: (n_samples, n_dims)
            y: (n_samples,)
        """
        seed_all(self.random_state)
        n_iter = n_iter or self.n_iter
        
        # Preprocess
        X = preprocess(X)
        X, y = sample_Xy(X, y, self.sample_num)
        self.variables = [nd.Variable(var) for var in X]
        self.keep_vars = len(X) if self.keep_vars and len(X) <= self.max_var / 2 else 0

        # Root Node
        variables = list(X.keys())
        if len(variables) > self.max_var: variables = variables[:self.max_var]
        self.MC_tree = Node([nd.Variable(var) for var in variables])
        self.estimate_MDL(self.MC_tree, X, y)

        # Search
        stop = False
        self.best = None
        self.start_time = time.time()
        for iter in tqdm(range(1, n_iter+1), disable=not use_tqdm):
            record = {'iter': iter, 'time': time.time() - self.start_time}
            log = {'Iter': iter}

            leaf = self.select(self.MC_tree)
            expand = self.expand(leaf, X, y)
            reward, best_simulated = self.simulate(expand, X, y)
            self.backpropagate(expand, reward)
            
            self.step_timer.add(1)

            if self.best is None or best_simulated.reward > self.best.reward:
                self.best = best_simulated.copy()
                self.set_reward(self.best, X, y)
                # original behavior: always simplify
                self.eqtree = simplify(self.best.phi)
                self._register_global_candidate(self.best)
                record['complexity'] = self.best.complexity
                record['reward'] = self.best.reward
                record['r2'] = self.best.r2
                record['mse'] = self.best.mse
                record['MDL'] = self.best.MDL
                record['eqtree'] = str(self.best)
                stop = early_stop(self.best.r2, self.best.complexity, self.best.phi)

            # Decide whether to log this iteration. Important: make time-based logging periodic
            # by resetting the step timer after a time-triggered log, otherwise it would fire on every
            # subsequent iteration once the threshold is exceeded.
            log_by_iter = bool(self.log_every_n_iters and (iter % self.log_every_n_iters == 0))
            # Keep backward-compatibility with the old "not iter % ..." behavior; treat as modulo trigger
            log_by_mod = bool(self.log_per_iter and ((iter % self.log_per_iter) == 0))
            log_by_time = bool(self.step_timer.time > self.log_per_sec)
            do_log = log_by_iter or log_by_mod or log_by_time or (iter == n_iter) or stop
            if do_log:
                record['speed'] = (str(self.step_timer), str(self.view_timer))
                record['detailed_time'] = str(self.named_timer)
                
                log['Reward'] = f'{self.best.reward:.5f}'
                log['Complexity'] = self.best.complexity
                log['R2'] = f'{self.best.r2:.5f}'
                if self.best.mse is not None:
                    log['MSE'] = f'{self.best.mse:.5f}'
                log['MDL'] = f'{self.best.MDL:.5f}'
                log['Best'] = str(self.best)
                log['Best equation'] = str(self.best.phi)
                log['Speed'] = f"{record['speed'][0]} ({record['speed'][1]})"
                log['Time'] = record['detailed_time']
                log['Current'] = str(expand)
                self.logger.info(' | '.join(f'\033[4m{k}\033[0m: {v}' for k, v in log.items()))
                # Reset the step timer if we logged due to time threshold to enforce periodic time-based logs
                if log_by_time:
                    self.step_timer.clear(reset=True)

            self.records.append(record)
            if self.save_path:
                with open(self.save_path, 'a') as f:
                    f.write(json.dumps(record) + '\n')

            if stop:
                self.logger.note(f'Early stopping at iter {iter} with R2 {self.best.r2} ({self.best.eqtrees})')
                break
        
        # Final Optim
        if self.calculator is not None and is_final_optim:
            top_nodes = self._get_global_top_k_nodes(k=5)
            mse_list = []
            for node_opt in top_nodes:
                Z = np.zeros((y.shape[0], 1+len(node_opt.eqtrees)))
                Z[:, 0] = 1.0
                for idx_t, eqtree in enumerate(node_opt.eqtrees, 1):
                    try:
                        Z[:, idx_t] = eqtree.eval(X)
                    except Exception:
                        Z[:, idx_t] = np.nan
                Z[~np.isfinite(Z)] = 0
                A_ls, _, _, _ = np.linalg.lstsq(Z, y, rcond=None)
                init_params = np.array(A_ls)
                for i in range(max_iter):
                    og_calc = copy.deepcopy(self.calculator)
                    sol = self.optimization(node_opt, init_params=init_params)
                    self.calculator = copy.deepcopy(og_calc)
                    new_eqtrees = []
                    new_params = [sol['x'][0]]
                    cnt = 0
                    for j in range(1, len(sol['x'])):
                        if abs(sol['x'][j]) >= 1e-5:
                            new_eqtrees.append(node_opt.eqtrees[j-1])
                            new_params.append(sol['x'][j])
                            cnt += 1
                    if cnt == len(sol['x']):
                        i = max_iter - 1  # early stop
                    node_opt.eqtrees = new_eqtrees
                    init_params = np.array(new_params)
                    if i == max_iter - 1:
                        h = self._compute_expression_hash(node_opt)
                        self.optimization_cache[h] = {
                            'sol': sol,
                            'r2': 1 / (1 + sol['fun']),
                            'mse': node_opt.mse,
                            'A': init_params
                        }
                        print(init_params, node_opt.mse)
                        mse_list.append(node_opt.mse)
            
            best_mse = float(np.nanmin(mse_list))
            self.logger.info(
                f"[final-opt] global top-10 optimized MSEs (min={best_mse:.3e}): "
                + ', '.join(f"{m:.3e}" for m in mse_list)
            )
            
            return best_mse

        return -self.best.r2
        
    def predict(self, X:np.ndarray|pd.DataFrame|Dict[str,np.ndarray]) -> np.ndarray:
        """
        Args:
            X: (n_samples, n_dims)
        Returns:
            y: (n_samples,)
        """
        if self.eqtree is None: raise ValueError('Model not fitted yet')
        X = preprocess(X)
        pred = self.eqtree.eval(X)
        pred[~np.isfinite(pred)] = 0
        return pred

    def action(self, state:Node, action:Tuple[nd.Symbol,int]) -> Node:
        """
        用 action[0] 取代 state.eqtrees[action[1]]
        """
        state = state.copy()
        eqtree, idx = action
        if idx == len(state.eqtrees): 
            state.eqtrees.append(eqtree)
        elif isinstance(eqtree, nd.Empty):
            state.eqtrees.pop(idx)
        else:
            state.eqtrees[idx] = eqtree
        return state

    def check_valid_action(self, state:Node, action:Tuple[nd.Symbol,int]) -> bool:
        eqtree, idx = action
        if idx > min(len(state.eqtrees) + 1, self.max_var): return False
        if idx == len(state.eqtrees) and isinstance(eqtree, nd.Empty): return False
        if len(state.eqtrees) == 1 and isinstance(eqtree, nd.Empty): return False
        if sum(len(eqtree) for i, eqtree in enumerate(state.eqtrees) if i != idx) + len(eqtree) > self.max_len: return False
        if idx < self.keep_vars: return False
        
        # 检查嵌套深度限制
        if self.max_nesting_depth is not None:
            nesting_depth = calculate_nesting_depth(eqtree)
            if nesting_depth > self.max_nesting_depth:
                return False
        
        # 检查是否已存在相同的表达式（避免重复项如 b1*gamma + b2*gamma）
        if not isinstance(eqtree, nd.Empty):
            # 将表达式转换为字符串进行比较
            new_expr_str = eqtree.to_str() if hasattr(eqtree, 'to_str') else str(eqtree)
            for i, existing_eqtree in enumerate(state.eqtrees):
                if i != idx:  # 不与自己比较
                    existing_expr_str = existing_eqtree.to_str() if hasattr(existing_eqtree, 'to_str') else str(existing_eqtree)
                    if new_expr_str == existing_expr_str:
                        # print(f"[MCTS DEBUG] 阻止重复表达式: {new_expr_str}")  # 可以取消注释来查看被阻止的重复项
                        return False
        
        # 检查并阻止无意义的表达式（如 x/x, x-x, x*1, x+0 等）
        if not isinstance(eqtree, nd.Empty) and hasattr(eqtree, 'operands') and eqtree.operands:
            if len(eqtree.operands) == 2:  # 二元操作
                left, right = eqtree.operands
                left_str = left.to_str() if hasattr(left, 'to_str') else str(left)
                right_str = right.to_str() if hasattr(right, 'to_str') else str(right)
                
                # 检查 x/x, x-x 类型的表达式
                if isinstance(eqtree, (nd.Div, nd.Sub)) and left_str == right_str:
                    # print(f"[MCTS DEBUG] 阻止无意义表达式: {left_str} {type(eqtree).__name__} {right_str}")
                    return False
                
                # 检查 x*1, 1*x 类型的表达式 (乘以1是冗余的)
                if isinstance(eqtree, nd.Mul):
                    if (isinstance(left, nd.Number) and abs(left.value - 1.0) < 1e-10) or \
                       (isinstance(right, nd.Number) and abs(right.value - 1.0) < 1e-10):
                        # print(f"[MCTS DEBUG] 阻止乘以1的表达式: {left_str} * {right_str}")
                        return False
                
                # 检查 x+0, 0+x 类型的表达式 (加0是冗余的)
                if isinstance(eqtree, nd.Add):
                    if (isinstance(left, nd.Number) and abs(left.value) < 1e-10) or \
                       (isinstance(right, nd.Number) and abs(right.value) < 1e-10):
                        # print(f"[MCTS DEBUG] 阻止加0的表达式: {left_str} + {right_str}")
                        return False
                
                # 检查 x*0, 0*x 类型的表达式 (乘以0结果为0，通常不希望出现)
                if isinstance(eqtree, nd.Mul):
                    if (isinstance(left, nd.Number) and abs(left.value) < 1e-10) or \
                       (isinstance(right, nd.Number) and abs(right.value) < 1e-10):
                        # print(f"[MCTS DEBUG] 阻止乘以0的表达式: {left_str} * {right_str}")
                        return False
        
        return True

    def iter_valid_action(self, state:Node, shuffle=False) -> Generator[Tuple[nd.Symbol,int],None,None]:
        leafs = [*state.eqtrees, *self.variables, *self.leaf]

        eqtree_loader = []
        for sym in self.binary:
            if sym in [nd.Add, nd.Mul, nd.Max, nd.Min]: # Abelian group
                for i in range(len(leafs)):
                    for j in range(i, len(leafs)):
                        eqtree_loader.append(sym(leafs[i], leafs[j]))
            elif sym in [nd.Sub, nd.Div]: # Non-abelian group
                for i in range(len(leafs)):
                    for j in range(len(leafs)):
                        if i != j:
                            eqtree_loader.append(sym(leafs[i], leafs[j]))
            else:
                for i in range(len(leafs)):
                    for j in range(len(leafs)):
                        eqtree_loader.append(sym(leafs[i], leafs[j]))
        for sym in self.unary:
            for i in range(len(leafs)):
                eqtree_loader.append(sym(leafs[i]))
        for sym in self.variables:
            eqtree_loader.append(sym)
        eqtree_loader.append(nd.Empty())

        idx_loader = list(range(self.keep_vars, min(len(state.eqtrees) + 1, self.max_var)))

        loader = [(eqtree, idx) for eqtree in eqtree_loader for idx in idx_loader]
        if shuffle:
            random.shuffle(loader)

        for eqtree, idx in loader:
            if self.check_valid_action(state, (eqtree, idx)):
                yield eqtree, idx
    
    def pick_valid_action(self, state:Node) -> Tuple[nd.Symbol,int]:
        leafs = [*state.eqtrees, *self.variables, *self.leaf]
        for _ in range(1000):
            op = random.choice(self.binary + self.unary + self.variables + [nd.Empty()])
            idx = random.choice(range(self.keep_vars, min(len(state.eqtrees) + 1, self.max_var)))
            if isinstance(op, type): op = op(*random.choices(leafs, k=op.n_operands))
            if self.check_valid_action(state, (op, idx)): break
        else:
            raise ValueError('Cannot find valid action')
        return op, idx

    def estimate_MDL(self, nodes:Node|List[Node], X:Dict[str,np.ndarray], y:np.ndarray, batch_size=64) -> float:
        if isinstance(nodes, Node): nodes = [nodes]
        batch = np.zeros((len(nodes), y.shape[0], self.max_var+1)) # (B, N_i, D_max+1,)
        for idx, node in enumerate(nodes):
            for i, eqtree in enumerate(node.eqtrees): 
                batch[idx, :, i] = eqtree.eval(X)
        if self.normalize_y: y = (y - y.mean()) / (y.std()+1e-6)
        batch[:, :, -1] = y[np.newaxis, :]
        if self.remove_abnormal: batch[((batch > -5) & (batch < 5)).all(axis=-1), :] = np.nan
        if self.normalize_all: batch = (batch - batch.mean(axis=1, keepdims=True)) / (batch.std(axis=1, keepdims=True)+1e-6)
        x = torch.from_numpy(self.tokenizer.float2index(batch, nan_to_pad=True)).to(self.model.device)
        pred = []
        for i in range(0, len(nodes), batch_size):
            pred.extend(self.model.predict(x[i:i+batch_size]).cpu().tolist())
        for node, c in zip(nodes, pred):
            node.MDL = c
        return pred if len(nodes) > 1 else pred[0]

    def select(self, root:Node) -> Node:
        node = root
        while node.children:
            node = max(node.children, key=lambda x: x.Q/(x.N+1e-6) + self.c / max(x.MDL, 1e-6) * np.sqrt(node.N) / (x.N+1))
        return node

    def expand(self, node:Node, X:Dict[str,np.ndarray], y:np.ndarray) -> Node:
        for idx, action in enumerate(self.iter_valid_action(node, shuffle=True)):
            child = self.action(node, action)
            child.parent = node
            child.xchild = len(node.children)
            node.children.append(child)
            if self.child_num and idx + 1 >= self.child_num: break
        if not node.children: return node  # leaf node
        self.estimate_MDL(node.children, X, y)
        return random.choice(node.children)

    def simulate(self, node:Node, X:Dict[str,np.ndarray], y:np.ndarray) -> Tuple[Node, float]:
        self.set_reward(node, X, y)
        self._register_global_candidate(node)
        best = node
        for i in range(self.n_playout):
            state = node
            for j in range(self.d_playout):
                action = self.pick_valid_action(state)
                if action is None: break
                state = self.action(state, action)
                self.set_reward(state, X, y)
                self._register_global_candidate(state)
                if state.reward > best.reward: best = state
        return best.reward, best

    def backpropagate(self, node:Node, reward:float):
        while node:
            node.N += 1
            node.Q += reward
            node = node.parent
            
    def optimization(self, node:Node, init_params:np.ndarray|None=None):
        # Convert an nd2py expression tree to a readable string.
        # Prefer the built-in to_str; fallback to a minimal recursive mapping.
        def transfer_nd_expr_to_str(expr, number_format: str = ".6g") -> str:
            if expr is None:
                return "0"
            if hasattr(expr, "to_str"):
                return expr.to_str(number_format=number_format)
            if isinstance(expr, nd.Number):
                return str(expr.value)
            if isinstance(expr, nd.Variable):
                return expr.name
            raise ValueError(f'Cannot convert {expr} to str')

        # 1) Build a linear template: y ≈ b0 + Σ bi * fi(x)
        fi_strs = [transfer_nd_expr_to_str(eq) for eq in node.eqtrees]

        # If self.x_name is provided (a list), remap variable names inside fi_strs
        # to the calculator's expected variable names. Map by order of self.variables.
        if self.x_name is not None and isinstance(self.x_name, (list, tuple)):
            src_names = [v.name for v in self.variables]
            # build mapping: src -> dst (only for indices available)
            name_map = {}
            for i, dst in enumerate(self.x_name):
                if i < len(src_names):
                    name_map[src_names[i]] = dst
            if name_map:
                # perform whole-word replacements
                def replace_vars(s: str) -> str:
                    for src, dst in name_map.items():
                        s = re.sub(rf"\b{re.escape(src)}\b", dst, s)
                    return s
                fi_strs = [replace_vars(s) for s in fi_strs]
        n_coef = 1 + len(fi_strs)
        coef_names = ["b0"] + [f"b{i}" for i in range(1, n_coef)]

        # 2) Register unknown coefficients to the calculator
        for name in coef_names:
            self.calculator.register_unknown_var(name)

        # 3) Compose expression string with those coefficients
        expr_terms = [coef_names[0]] + [f"{coef_names[i+1]}*({fi})" for i, fi in enumerate(fi_strs)]
        expr_str = " + ".join(expr_terms) if expr_terms else "0"
        self.calculator.update_unknown_var(self.y_name, expr_str)
        self.calculator.upd_local_dict()
        train_loss_func, mse_list_train = self.calculator.get_loss_func(deci_list_len=self.deci_list_len)
        unk_list = list(self.calculator.sp_unknown_quantities.keys())
        n_params = len(unk_list)
        if init_params is None or len(init_params) != n_params:
            init_params = np.random.rand(n_params)
    
        sol = optimize_with_timeout(
            train_loss_func,
            init_params,
            self.calculator.get_constr_dict_list(),
            prev_sol_best=None,
            verbose=True,
        )
        node.mse = sol['fun']
        return sol

    def set_reward(self, node:Node, X:Dict[str,np.ndarray], y:np.ndarray) -> float:
        self.view_timer.add(1)
        if self.train_eval_split < 1.0:
            train_idx = np.random.rand(y.shape[0]) < self.train_eval_split
            eval_idx = ~train_idx
        else:
            train_idx = np.ones_like(y).astype(bool)
            eval_idx = train_idx
        if isinstance(self.other_params_name, list) and self.y_name in self.other_params_name:
            self.other_params_name.remove(self.y_name)
        # linear model as phi
        if self.mode == 'optimization':
            # 检查缓存，避免重复优化相同的表达式
            expr_hash = self._compute_expression_hash(node)
            if expr_hash in self.optimization_cache:
                cached_result = self.optimization_cache[expr_hash]
                sol = cached_result['sol']
                node.r2 = cached_result['r2']
                node.mse = cached_result['mse']
                A = cached_result['A']
                print(f"[CACHE HIT] 使用缓存的优化结果，hash: {expr_hash[:8]}...")
            else:
                og_calc = copy.deepcopy(self.calculator)
                if self.other_params_name and len(self.other_params_name) > 0:
                    raise NotImplementedError('other_params is not supported yet')
                sol = self.optimization(node)
                node.r2 = 1 / (1 + sol['fun'])    # Not r2, actually MSE loss
                try:
                    print("Total MSE: ", sol['fun'])
                    A = np.array(sol['x'])
                    self.optimization_cache[expr_hash] = {
                        'sol': sol,
                        'r2': node.r2,
                        'mse': node.mse,
                        'A': A
                    }
                except:
                    raise ValueError('Optimization failed')
                    
                # print(sol['x'])
                self.calculator = copy.deepcopy(og_calc)
        else:
            # Calculate Z
            Z = np.zeros((y.shape[0], 1+len(node.eqtrees)))
            Z[:, 0] = 1.0
            for idx, eqtree in enumerate(node.eqtrees, 1):
                try:
                    Z[:, idx] = eqtree.eval(X)
                except:
                    Z[:, idx] = np.nan
            Z[~np.isfinite(Z)] = 0
            A, _, _, _ = np.linalg.lstsq(Z[train_idx, :], y[train_idx], rcond=None)
            A = np.round(A, 6)
            node.r2 = R2_score(y[eval_idx], Z[eval_idx, :] @ A)
            node.mse = None
        
        node.phi = nd.Number(A[0]) if A[0] != 0 else None
        for a, op in zip(A[1:], node.eqtrees):
            if a == 0: pass
            elif a == 1: 
                if node.phi is None: node.phi = op
                else: node.phi += op
            elif a == -1:
                if node.phi is None: node.phi = -op
                else: node.phi -= op
            else: 
                if node.phi is None: node.phi = nd.Number(a) * op
                else: node.phi += nd.Number(a) * op
        if node.phi is None: node.phi = nd.Number(0.0)
        node.complexity = len(node.phi)
        penalty_complexity = node.complexity
        if self.complexity_limit is not None and penalty_complexity > self.complexity_limit:
            node.r2 = -np.inf
            node.reward = 0.0
            return 0.0
        node.reward = self.eta ** (self.complexity_alpha * penalty_complexity) / (2 - node.r2)

        # prod model as phi: y = phi(f1, f2, ...) = a0 * |f1|^a1 * |f2|^a2 * ...
        try:
            if self.disable_prod_model:
                raise ValueError('prod_model_disabled')  # handled below silently
            Z_ = np.log(np.abs(Z).clip(1e-10, None))
            Z_[:, 0] = 1.0
            y_ = np.log(np.abs(y).clip(1e-10, None))
            assert np.isfinite(Z_).all() and np.isfinite(y_).all()
            A, _, _, _ = np.linalg.lstsq(Z_[train_idx, :], y_[train_idx], rcond=None)
            A[0] = np.exp(A[0])
            A = np.round(A, 6)
            # clip powers if requested
            if self.max_power_abs is not None:
                for i in range(1, len(A)):
                    if np.isfinite(A[i]):
                        if A[i] > self.max_power_abs: A[i] = self.max_power_abs
                        elif A[i] < -self.max_power_abs: A[i] = -self.max_power_abs
            prod = 1
            for z, a in zip(Z[:, 1:].T, A[1:]):
                prod *= np.abs(z) ** a
            A[0] *= np.sign(np.median(y[train_idx] / (A[0] * prod[train_idx]).clip(1e-6)))
            r2 = R2_score(y[eval_idx], A[0] * prod[eval_idx])
            penalty_complexity = node.complexity
            if self.complexity_limit is not None and penalty_complexity > self.complexity_limit:
                raise ValueError('complexity_overflow_prod')
            reward = self.eta ** (self.complexity_alpha * penalty_complexity) / (2 - r2)
            if reward > node.reward:
                node.r2 = r2
                node.reward = reward
                node.phi = nd.Number(A[0]) if A[0] != 1 else None
                for idx, (a, op) in enumerate(zip(A[1:], node.eqtrees), 1):
                    if (Z[idx]<0).any(): op = nd.Abs(op)
                    if a == 0: pass
                    elif a == 1: 
                        if node.phi is None: node.phi = op
                        else: node.phi *= op
                    elif a == -1: 
                        if node.phi is None: node.phi = nd.Inv(op)
                        else: node.phi /= op
                    elif a == 2:
                        if node.phi is None: node.phi = nd.Pow2(op)
                        else: node.phi *= nd.Pow2(op)
                    elif a == -2:
                        if node.phi is None: node.phi = nd.Inv(nd.Pow2(op))
                        else: node.phi /= nd.Pow2(op)
                    elif a == 3:
                        if node.phi is None: node.phi = nd.Pow3(op)
                        else: node.phi *= nd.Pow3(op)
                    elif a == -3:
                        if node.phi is None: node.phi = nd.Inv(nd.Pow3(op))
                        else: node.phi /= nd.Pow3(op)
                    elif a == 0.5:
                        if node.phi is None: node.phi = nd.Sqrt(op)
                        else: node.phi *= nd.Sqrt(op)
                    elif a == -0.5:
                        if node.phi is None: node.phi = nd.Inv(nd.Sqrt(op))
                        else: node.phi /= nd.Sqrt(op)
                    elif a > 0:
                        if node.phi is None: node.phi = op ** nd.Number(a)
                        else: node.phi *= op ** nd.Number(a)
                    elif a < 0:
                        if node.phi is None: node.phi = nd.Inv(op ** nd.Number(-a))
                        else: node.phi /= op ** nd.Number(-a)
                    else: raise ValueError(f'Unknown a: {a}')
                if node.phi is None: node.phi = nd.Number(1.0)
                node.complexity = len(node.phi)
        except Exception as e:
            # Suppress verbosity for deliberate disables / overflow
            if 'prod_model_disabled' not in str(e) and 'complexity_overflow_prod' not in str(e):
                self.logger.warning(traceback.format_exc())
            node.r2 = node.r2 if node.r2 is not None else -np.inf
            # keep existing linear reward; do not overwrite if already set
        
        if not np.isfinite(node.reward): node.reward = 0.0
