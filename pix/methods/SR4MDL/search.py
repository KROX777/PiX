import os
import json
import yaml
import time
import torch
import signal
import logging
import traceback
import numpy as np
import nd2py as nd2
import pandas as pd
from socket import gethostname
from argparse import ArgumentParser
from setproctitle import setproctitle
from nd2py.utils import seed_all, init_logger, AutoGPU, AttrDict
from sr4mdl.utils import parse_parser, RMSE_score, R2_score
from sr4mdl.search import MCTS4MDL
from sr4mdl.model import MDLformer
from sr4mdl.env import sympy2eqtree, str2sympy, Tokenizer


parser = ArgumentParser()
parser.add_argument('-f', '--function', type=str, default='f=x1+x2*sin(x3)', help='`f=...\' or `Feynman_xxx\'')
parser.add_argument('-n', '--name', type=str, default=None)
parser.add_argument('-s', '--seed', type=int, default=0)
parser.add_argument('--device', type=str, default='auto')
parser.add_argument('--sample_num', type=int, default=200)
parser.add_argument('--c', type=float, default=1.41)
parser.add_argument('--max_len', type=int, default=30)
parser.add_argument('--n_iter', type=int, default=10000)
parser.add_argument('--max_var', type=int, default=10)
parser.add_argument('--load_model', type=str, default='./weights/checkpoint.pth')
parser.add_argument('--quiet', action='store_true')
parser.add_argument('--keep_vars', action='store_true')
parser.add_argument('--normalize_y', action='store_true')
parser.add_argument('--normalize_all', action='store_true')
parser.add_argument('--remove_abnormal', action='store_true')
parser.add_argument('--use_old_model', action='store_true')
parser.add_argument('--cheat', action='store_true')
parser.add_argument('--domain', type=str, default='-5,5',
                    help="Sampling domain. Format: 'low,high' applied to all vars or 'x1:low,high;x2:low,high'.")
parser.add_argument('--avoid_zero_eps', type=float, default=0.0,
                    help='If >0, resample any sampled value with |x| < eps (avoid log/division issues).')
parser.add_argument('--eta', type=float, default=0.999, help='Complexity decay factor in reward.')
parser.add_argument('--child_num', type=int, default=50, help='Max children expanded per node.')
parser.add_argument('--n_playout', type=int, default=100, help='Number of playout simulations per expansion.')
parser.add_argument('--d_playout', type=int, default=10, help='Depth per playout.')
parser.add_argument('--extra_leaf', type=str, default='', help='Extra numeric constants, comma separated, e.g. "0.01,0.1,10"')
parser.add_argument('--disable_prod_model', action='store_true', help='Disable multiplicative (prod/exp) model fitting.')
parser.add_argument('--max_power_abs', type=float, default=None, help='Clip absolute value of exponents in product model.')
parser.add_argument('--complexity_limit', type=int, default=None, help='Hard cap: expressions with complexity above get zero reward.')
parser.add_argument('--complexity_alpha', type=float, default=1.0, help='Scale complexity inside eta exponent: eta^(alpha*complexity).')
parser.add_argument('--custom_binary_ops', type=str, default='', 
                    help='Manually specify binary operators, comma separated. e.g. "Add,Mul,Sub,Div". Available: Add,Sub,Mul,Div,Pow')
parser.add_argument('--custom_unary_ops', type=str, default='', 
                    help='Manually specify unary operators, comma separated. e.g. "Sin,Cos,Log,Exp". Available: Sin,Cos,Tan,Log,Exp,Sqrt,Inv,Neg,Pow2,Pow3,Arcsin,Arccos,Cot,Tanh')
parser.add_argument('--custom_leaf_ops', type=str, default='', 
                    help='Manually specify leaf constants, comma separated. e.g. "1,2,3.14159,2.718". Use "pi" for π, "e" for Euler number')
parser.add_argument('--use_custom_ops_only', action='store_true', 
                    help='If set, use only the manually specified operators (ignore auto-detected ones)')


args = parse_parser(parser, save_dir='./results/search/')

init_logger('sr4mdl', args.name, os.path.join(args.save_dir, 'info.log'))
logger = logging.getLogger('sr4mdl.search')
logger.info(args)
seed_all(args.seed)
def handler(signum, frame): raise KeyboardInterrupt
signal.signal(signal.SIGINT, handler)
signal.signal(signal.SIGTERM, handler)
setproctitle(f'{args.name}@YuZihan')

if args.device == 'auto':
    args.device = AutoGPU().choice_gpu(memory_MB=1486, interval=15)
args.function = args.function.replace(' ', '')


def parse_custom_operators(custom_binary_ops='', custom_unary_ops='', custom_leaf_ops=''):
    """Parse custom operator specifications into nd2py operator classes"""
    
    # Available operator mappings
    BINARY_OPS = {
        'Add': nd2.Add, 'Sub': nd2.Sub, 'Mul': nd2.Mul, 'Div': nd2.Div, 'Pow': nd2.Pow
    }
    
    UNARY_OPS = {
        'Sin': nd2.Sin, 'Cos': nd2.Cos, 'Tan': nd2.Tan,
        'Log': nd2.Log, 'Exp': nd2.Exp, 'Sqrt': nd2.Sqrt,
        'Inv': nd2.Inv, 'Neg': nd2.Neg, 'Pow2': nd2.Pow2, 'Pow3': nd2.Pow3,
        'Arcsin': nd2.Arcsin, 'Arccos': nd2.Arccos, 'Cot': nd2.Cot, 'Tanh': nd2.Tanh
    }
    
    # Parse binary operators
    binary_ops = []
    if custom_binary_ops.strip():
        for op_name in custom_binary_ops.split(','):
            op_name = op_name.strip()
            if op_name in BINARY_OPS:
                binary_ops.append(BINARY_OPS[op_name])
            else:
                logger.warning(f"Unknown binary operator: {op_name}. Available: {list(BINARY_OPS.keys())}")
    
    # Parse unary operators
    unary_ops = []
    if custom_unary_ops.strip():
        for op_name in custom_unary_ops.split(','):
            op_name = op_name.strip()
            if op_name in UNARY_OPS:
                unary_ops.append(UNARY_OPS[op_name])
            else:
                logger.warning(f"Unknown unary operator: {op_name}. Available: {list(UNARY_OPS.keys())}")
    
    # Parse leaf constants
    leaf_ops = []
    if custom_leaf_ops.strip():
        for leaf_val in custom_leaf_ops.split(','):
            leaf_val = leaf_val.strip()
            try:
                if leaf_val.lower() == 'pi':
                    leaf_ops.append(nd2.Number(np.pi))
                elif leaf_val.lower() == 'e':
                    leaf_ops.append(nd2.Number(np.e))
                else:
                    val = float(eval(leaf_val, {"__builtins__": {}}, {"pi": np.pi, "e": np.e}))
                    leaf_ops.append(nd2.Number(val))
            except Exception as e:
                logger.warning(f"Failed to parse leaf constant '{leaf_val}': {e}")
    
    return binary_ops, unary_ops, leaf_ops


def search():
    if '=' in args.function:
        f = sympy2eqtree(str2sympy(args.function.split('=', 1)[1]))
        # Collect operators / leaves (Numbers are unhashable so we deduplicate manually)
        binary = []
        unary = []
        variables = []
        leaf = []
        seen_binary = set()
        seen_unary = set()
        seen_vars = set()
        seen_leaf = set()
        for op in f.iter_preorder():
            if op.n_operands == 2:
                if op.__class__ not in seen_binary:
                    seen_binary.add(op.__class__)
                    binary.append(op.__class__)
            elif op.n_operands == 1:
                if op.__class__ not in seen_unary:
                    seen_unary.add(op.__class__)
                    unary.append(op.__class__)
            if isinstance(op, nd2.Variable):
                if op.name not in seen_vars:
                    seen_vars.add(op.name)
                    variables.append(op.name)
            if isinstance(op, nd2.Number):
                # use string format as key to avoid float precision mismatch
                k = op.to_str(number_format=".8f")
                if k not in seen_leaf:
                    seen_leaf.add(k)
                    leaf.append(op)
        # fallback: ensure at least a constant 1 exists
        if not leaf:
            leaf.append(nd2.Number(1))
        # --- domain parsing & sampling ---
        def _parse_domain(domain_str, vars_):
            mapping = {}
            if ';' in domain_str or ':' in domain_str:
                for seg in domain_str.split(';'):
                    seg = seg.strip()
                    if not seg:
                        continue
                    if ':' not in seg:
                        raise ValueError(f"Segment '{seg}' missing ':' (expected var:low,high)")
                    var, rng = seg.split(':', 1)
                    var = var.strip()
                    low_high = [r.strip() for r in rng.split(',')]
                    if len(low_high) != 2:
                        raise ValueError(f"Range '{rng}' must be 'low,high'")
                    low, high = map(float, low_high)
                    mapping[var] = (low, high)
            else:
                low_high = [r.strip() for r in domain_str.split(',')]
                if len(low_high) != 2:
                    raise ValueError("Domain must be 'low,high' or 'v1:low,high;v2:low,high'")
                low, high = map(float, low_high)
                for v in vars_:
                    mapping[v] = (low, high)
            for v in vars_:
                if v not in mapping:
                    mapping[v] = (-5.0, 5.0)
            return mapping

        try:
            domain_map = _parse_domain(args.domain, variables)
        except Exception as e:
            logger.warning(f"Failed to parse domain '{args.domain}': {e}; fallback to (-5,5)")
            domain_map = {v: (-5.0, 5.0) for v in variables}

        X = {}
        for var in variables:
            low, high = domain_map[var]
            vals = np.random.uniform(low, high, (args.sample_num,))
            if args.avoid_zero_eps > 0:
                mask = np.abs(vals) < args.avoid_zero_eps
                attempts = 0
                while mask.any() and attempts < 10:
                    vals[mask] = np.random.uniform(low, high, mask.sum())
                    mask = np.abs(vals) < args.avoid_zero_eps
                    attempts += 1
            X[var] = vals
        y = f.eval(X)
        if not np.isfinite(y).all():
            finite_mask = np.isfinite(y)
            for v in X:
                X[v] = X[v][finite_mask]
            y = y[finite_mask]
            old_n = args.sample_num
            args.sample_num = y.shape[0]
            logger.warning(f"Filtered non-finite samples: {old_n} -> {args.sample_num}")
        log = {
            'target function': args.function,
            'binary operators': [op.__name__ for op in binary],
            'unary operators': [op.__name__ for op in unary],
            'leaf': [op.to_str(number_format=".2f") for op in leaf],
            'variables': list(X.keys()),
            'domain': {k: list(v) for k, v in domain_map.items()} if variables else {}
        }
    else:
        import pmlb
        logger.info(f'fetching {args.function} from PMLB...')
        os.makedirs('./data/cache', exist_ok=True)
        df = pmlb.fetch_data(args.function, local_cache_dir='./data/cache/')
        if df.shape[0] > args.sample_num: 
            df = df.sample(args.sample_num)
        else: 
            args.sample_num = df.shape[0]
        logger.info(f'Done, df.shape = {df.shape}')
        X = {col:df[col].values for col in df.columns}
        y = X.pop('target')
        binary = [nd2.Mul, nd2.Div, nd2.Add, nd2.Sub]
        unary = [nd2.Sqrt, nd2.Cos, nd2.Sin, nd2.Pow2, nd2.Pow3, nd2.Exp, nd2.Inv, nd2.Neg, nd2.Arcsin, nd2.Arccos, nd2.Cot, nd2.Log, nd2.Tanh]
        leaf = [nd2.Number(1), nd2.Number(2), nd2.Number(np.pi)]
        log = {
            'target function': args.function,
            'binary operators': [op.__name__ for op in binary],
            'unary operators': [op.__name__ for op in unary],
            'leaf': [op.to_str(number_format=".2f") for op in leaf],
            'variables': list(X.keys()),
        }
        try:
            metadata = yaml.load(open(f'./data/pmlb/datasets/{args.function}/metadata.yaml', 'r'), Loader=yaml.Loader)['description']
            metadata = [l.strip() for l in metadata.split('\n')]
            target, eq = metadata[metadata.index('')+1].split(' = ', 1)
            eq = sympy2eqtree(str2sympy(eq))
            log['target function'] = log['target function'] + ' ({} = {})'.format(target, eq.to_str(number_format=".2f"))
            if args.cheat:
                # Recompute operators from ground-truth equation (cheat mode)
                binary, unary, leaf = [], [], []
                seen_binary = set(); seen_unary = set(); seen_leaf = set()
                for op in eq.iter_preorder():
                    if op.n_operands == 2 and op.__class__ not in seen_binary:
                        seen_binary.add(op.__class__); binary.append(op.__class__)
                    elif op.n_operands == 1 and op.__class__ not in seen_unary:
                        seen_unary.add(op.__class__); unary.append(op.__class__)
                    if isinstance(op, nd2.Number):
                        k = op.to_str(number_format=".8f")
                        if k not in seen_leaf:
                            seen_leaf.add(k); leaf.append(op)
                if not leaf: leaf.append(nd2.Number(1))
                log['binary operators'] = [op.__name__ for op in binary]
                log['unary operators'] = [op.__name__ for op in unary]
                log['leaf'] = [op.to_str(number_format=".2f") for op in leaf]
        except Exception as e:
            logger.warning(e)
    
    # Handle custom operators specification
    custom_binary, custom_unary, custom_leaf = parse_custom_operators(
        args.custom_binary_ops, args.custom_unary_ops, args.custom_leaf_ops
    )
    
    if args.use_custom_ops_only:
        # Use only manually specified operators
        if custom_binary:
            binary = custom_binary
            logger.note(f"Using custom binary operators only: {[op.__name__ for op in binary]}")
        if custom_unary:
            unary = custom_unary
            logger.note(f"Using custom unary operators only: {[op.__name__ for op in unary]}")
        if custom_leaf:
            leaf = custom_leaf
            logger.note(f"Using custom leaf constants only: {[op.to_str(number_format='.2f') for op in leaf]}")
    else:
        # Merge custom operators with existing ones
        if custom_binary:
            # Add custom binary operators (avoid duplicates)
            for op in custom_binary:
                if op not in binary:
                    binary.append(op)
            logger.note(f"Added custom binary operators: {[op.__name__ for op in custom_binary]}")
        
        if custom_unary:
            # Add custom unary operators (avoid duplicates)
            for op in custom_unary:
                if op not in unary:
                    unary.append(op)
            logger.note(f"Added custom unary operators: {[op.__name__ for op in custom_unary]}")
        
        if custom_leaf:
            # Add custom leaf constants (check for duplicates by value)
            for new_leaf in custom_leaf:
                is_duplicate = False
                for existing_leaf in leaf:
                    if isinstance(existing_leaf, nd2.Number) and isinstance(new_leaf, nd2.Number):
                        if abs(existing_leaf.value - new_leaf.value) < 1e-12:
                            is_duplicate = True
                            break
                if not is_duplicate:
                    leaf.append(new_leaf)
            logger.note(f"Added custom leaf constants: {[op.to_str(number_format='.2f') for op in custom_leaf]}")
    
    # Update log with final operator sets
    log['binary operators'] = [op.__name__ for op in binary]
    log['unary operators'] = [op.__name__ for op in unary]
    log['leaf'] = [op.to_str(number_format=".2f") for op in leaf]
    
    logger.note('\n'.join(f'{k}: {v if not isinstance(v, list) else "[" + ", ".join(v) + "]"}' for k, v in log.items()))

    tokenizer = Tokenizer(-100, 100, 4, args.max_var)
    state_dict = torch.load(args.load_model)
    model_args = AttrDict(dropout=0.1, d_model=512, d_input=64, d_output=512, n_TE_layers=8, max_len=50, max_param=5, max_var=args.max_var, uniform_sample_number=args.sample_num,device=args.device, use_SENet=True, use_old_model=args.use_old_model)
    model = MDLformer(model_args, state_dict['xy_token_list'])
    model.load(state_dict['xy_encoder'], state_dict['xy_token_list'], strict=True)
    model.eval()
    est = MCTS4MDL(
        tokenizer=tokenizer,
        model=model,
        n_iter=args.n_iter,
    c=args.c,
    sample_num=args.sample_num,
        keep_vars=args.keep_vars,
        normalize_y=args.normalize_y,
        normalize_all=args.normalize_all,
        remove_abnormal=args.remove_abnormal,
        binary=binary,
        unary=unary,
        leaf=leaf,
        child_num=args.child_num,
        n_playout=args.n_playout,
        d_playout=args.d_playout,
        log_per_sec=5,
        save_path=os.path.join(args.save_dir, 'records.json'),
        eta=args.eta,
        disable_prod_model=args.disable_prod_model,
        max_power_abs=args.max_power_abs,
        complexity_limit=args.complexity_limit,
        complexity_alpha=args.complexity_alpha,
    )

    # Inject extra leaf constants AFTER estimator build (no harm if duplicates) but before fit
    if args.extra_leaf:
        try:
            extras = []
            for tok in args.extra_leaf.split(','):
                tok = tok.strip()
                if not tok:
                    continue
                val = float(eval(tok, {}, {}))  # allow simple expressions like 1/100
                if all(abs(val - l.value) > 1e-12 if isinstance(l, nd2.Number) else True for l in est.leaf):
                    extras.append(nd2.Number(val))
            if extras:
                est.leaf.extend(extras)
                logger.note(f"Added extra leaf constants: {[x.to_str() for x in extras]}")
        except Exception as e:
            logger.warning(f"Failed parsing extra_leaf '{args.extra_leaf}': {e}")

    try:
        est.fit(X, y, use_tqdm=False)
        logger.info('Finished')
    except KeyboardInterrupt:
        logger.note('Interrupted')
    except Exception:
        logger.error(traceback.format_exc())

    y_pred = est.predict(X)
    rmse = RMSE_score(y, y_pred)
    r2 = R2_score(y, y_pred)
    # --- Debug: compare internal search R2 (est.best.r2) with raw eval R2 on same data ---
    internal_r2 = getattr(est.best, 'r2', None)
    internal_complexity = getattr(est.best, 'complexity', None)
    internal_reward = getattr(est.best, 'reward', None)
    if internal_r2 is not None:
        # recompute theoretical reward component to see mismatch
        try:
            recomputed_reward = (args.eta ** (args.complexity_alpha * internal_complexity)) / (2 - internal_r2)
        except Exception:
            recomputed_reward = None
        logger.note(
            f'Result = {est.eqtree}, RMSE = {rmse:.4f}, R2 = {r2:.4f} | Internal(best) R2={internal_r2:.5f}, complexity={internal_complexity}, '
            f'internal_reward={internal_reward:.5f} (recomputed={recomputed_reward})'
        )
    else:
        logger.note(f'Result = {est.eqtree}, RMSE = {rmse:.4f}, R2 = {r2:.4f}')

    result = {
        'date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'host': gethostname(),
        'name': args.name,
        'load_model': args.load_model,
        'success': str(rmse < 1e-6),
        'n_iter': len(est.records),
        'duration': est.records[-1]['time'],
        'model': 'MCTS4MDL',
        'exp': args.function,
        'result': str(est.eqtree),
        'rmse': rmse,
        'r2': r2,
        'sample_num': args.sample_num,
        'seed': args.seed,
    }
    json.dump(result, open(os.path.join(args.save_dir, 'result.json'), 'w'), indent=4)

    # aggregate results to aggregate.csv
    save_path = './results/aggregate.csv'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not os.path.exists(save_path):
        with open(save_path, 'w') as f:
            f.write('\t'.join([
                'success','name','exp','n_iter','duration','seed','rmse','r2',
                'result','target','date','host','load_model','model','sample_num'
            ]) + '\n')
    with open(save_path, 'a') as f:
        keys = open(save_path, 'r').readline().split('\t')
        f.write(','.join(str(result.get(k, '')) for k in keys) + '\n')


if __name__ == '__main__':
    search()
