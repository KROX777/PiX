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
import sys
import os as _os
from setproctitle import setproctitle
from nd2py.utils import seed_all, init_logger, AutoGPU, AttrDict
import importlib
import importlib.util
import types

# Ensure local `sr4mdl` package is importable even when not installed system-wide.
try:
    # try a normal import first
    import sr4mdl  # noqa: F401
except Exception:
    # If import fails, attempt to add the local SR4MDL folder to sys.path and retry
    _local_pkg_dir = os.path.join(os.path.dirname(__file__), 'sr4mdl')
    if os.path.isdir(_local_pkg_dir):
        if os.path.dirname(__file__) not in sys.path:
            sys.path.insert(0, os.path.dirname(__file__))
        try:
            import sr4mdl  # noqa: F401
        except Exception:
            # As a final fallback, synthesize a package module and load common submodules
            pkg = types.ModuleType('sr4mdl')
            pkg.__path__ = [_local_pkg_dir]
            sys.modules['sr4mdl'] = pkg
            # attempt to load subpackages used below: utils, search, model, env
            for sub in ('utils', 'search', 'model', 'env'):
                sub_path = os.path.join(_local_pkg_dir, sub)
                # if it's a package directory, ensure Python can find it via package __path__
                if os.path.isdir(sub_path):
                    # no-op: having sr4mdl in sys.modules with __path__ is usually enough
                    continue
                # if it's a single file module like utils.py, import it directly
                mod_file = os.path.join(_local_pkg_dir, f'{sub}.py')
                if os.path.isfile(mod_file):
                    spec = importlib.util.spec_from_file_location(f'sr4mdl.{sub}', mod_file)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    sys.modules[f'sr4mdl.{sub}'] = module

# Now perform the imports that expect a top-level `sr4mdl` package
from sr4mdl.utils import RMSE_score, R2_score
from sr4mdl.search import MCTS4MDL
from sr4mdl.model import MDLformer
from sr4mdl.env import sympy2eqtree, str2sympy, Tokenizer


config_path = os.path.join(os.path.dirname(__file__), 'cfg', 'config.yaml')

# temporary logger for config loading before init_logger sets up file handlers
logger = logging.getLogger('sr4mdl.search')
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# Load config file and build args (config keys provide defaults). Support JSON or YAML.
if not config_path:
    logger.error('No config path available. Set SR4MDL_CONFIG or provide a default config file.')
    raise SystemExit(1)
if not _os.path.exists(config_path):
    logger.error(f'Config file not found: {config_path}')
    raise SystemExit(1)
try:
    with open(config_path, 'r') as f:
        if config_path.lower().endswith('.json'):
            cfg = json.load(f)
        else:
            cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError('Config file must contain a mapping at top level')
except Exception as e:
    logger.error(f'Failed to load config file {config_path}: {e}')
    raise

# normalize cfg
if cfg is None:
    cfg = {}

# Build args from config (simple: use config values, fallback to timestamp/0)
args = AttrDict(cfg if cfg is not None else {})
args.name = args.get('name') or time.strftime('%Y%m%d_%H%M%S')
try:
    args.seed = int(args.get('seed', 0))
except Exception:
    args.seed = 0

# ensure save_dir exists
args.save_dir = os.path.join('./results/search/', args.name)
os.makedirs(args.save_dir, exist_ok=True)

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

def search(calculator=None, y_name=None, x_name=None, other_params_name=None, deci_list_len=None,
           X_override=None, y_override=None):
    # 1) Dataset override mode: if X_override and y_override are provided, use them directly.
    if X_override is not None and y_override is not None:
        # Accept dict-like X or numpy array; y as 1D array-like
        if isinstance(X_override, dict):
            X = {k: np.asarray(v) for k, v in X_override.items()}
            # Ensure all columns share the same length
            lengths = {k: len(v) for k, v in X.items()}
            if len(set(lengths.values())) != 1:
                raise ValueError(f"Inconsistent feature lengths: {lengths}")
            N = next(iter(lengths.values()))
        else:
            X_arr = np.asarray(X_override)
            if X_arr.ndim != 2:
                raise ValueError("X_override must be dict or 2D array [N, D]")
            N, D = X_arr.shape
            # auto-name variables x0,x1,...
            X = {f"x{i}": X_arr[:, i] for i in range(D)}
        y = np.asarray(y_override).reshape(-1)
        if y.shape[0] != N:
            raise ValueError(f"y_override length {y.shape[0]} != N {N}")

        # default operators if none deduced from args.function path; keep prior defaults
        binary = [nd2.Mul, nd2.Div, nd2.Add, nd2.Sub]
        unary = [nd2.Sqrt, nd2.Cos, nd2.Sin, nd2.Pow2, nd2.Pow3, nd2.Exp, nd2.Inv, nd2.Neg, nd2.Arcsin, nd2.Arccos, nd2.Cot, nd2.Log, nd2.Tanh]
        leaf = [nd2.Number(1), nd2.Number(2), nd2.Number(np.pi)]

        # respect sample_num as cap
        if N > args.sample_num:
            idx = np.random.choice(N, args.sample_num, replace=False)
            for k in X:
                X[k] = X[k][idx]
            y = y[idx]
            N = args.sample_num
        else:
            args.sample_num = N

        log = {
            'target function': 'dataset-override',
            'binary operators': [op.__name__ for op in binary],
            'unary operators': [op.__name__ for op in unary],
            'leaf': [op.to_str(number_format=".2f") for op in leaf],
            'variables': list(X.keys()),
            'domain': {},
            'N': int(N),
        }

    elif '=' in args.function:
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
        # Use only manually specified operators (even if empty)
        binary = custom_binary
        logger.note(f"Using custom binary operators only: {[op.__name__ for op in binary]}")
        unary = custom_unary
        logger.note(f"Using custom unary operators only: {[op.__name__ for op in unary]}")
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
    # Safely load checkpoint: if CUDA is not available, force mapping to CPU.
    model_path = os.path.join(os.path.dirname(__file__), args.load_model)
    try:
        if torch.cuda.is_available():
            map_location = args.device if hasattr(args, 'device') else None
        else:
            map_location = torch.device('cpu')
    except Exception:
        # Fallback: map to CPU if anything unexpected happens
        map_location = torch.device('cpu')
    # torch.load accepts either a device string/torch.device or None
    state_dict = torch.load(model_path, map_location=map_location)
    # Ensure model device is valid for this runtime: if CUDA is not available, force CPU
    try:
        if torch.cuda.is_available():
            safe_device = args.device
        else:
            logger.note('CUDA not available: forcing model device to CPU')
            safe_device = torch.device('cpu')
    except Exception:
        safe_device = torch.device('cpu')

    model_args = AttrDict(
        dropout=0.1,
        d_model=512,
        d_input=64,
        d_output=512,
        n_TE_layers=8,
        max_len=50,
        max_param=5,
        max_var=args.max_var,
        uniform_sample_number=args.sample_num,
        device=safe_device,
        use_SENet=True,
        use_old_model=args.use_old_model,
    )
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
        max_nesting_depth=args.get('max_nesting_depth'),
        calculator=calculator,
        y_name=y_name,
        x_name=x_name,
        other_params_name=other_params_name,
        deci_list_len=deci_list_len,
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
        'load_model': os.path.join(os.path.dirname(__file__), args.load_model),
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
