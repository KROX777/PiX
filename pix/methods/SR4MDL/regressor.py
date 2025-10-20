import torch
import numpy as np
from types import SimpleNamespace
from pathlib import Path

try:
    from nd2py.utils import init_logger, AutoGPU, AttrDict
    from sr4mdl.env import Tokenizer
    from sr4mdl.search import MCTS4MDL
    from sr4mdl.model import MDLformer
except Exception:
    # 延迟导入失败时允许最小功能 (回退到简易拟合)
    Tokenizer = None
    MCTS4MDL = None
    MDLformer = None
    AutoGPU = None
    AttrDict = dict

init_logger('sr4mdl', exp_name='regressor', info_level='note')

def load_cfg(path: str):
    """简单加载/构造 sr4mdl 配置。
    若路径不存在，返回默认配置对象，包含 Inference.weights_path 字段。
    """
    import yaml, os
    default = {
        'num_vars': 3,
        'Inference': {
            'weights_path': 'weights/checkpoint.pth',
            'n_points': 200,
            'use_gpu': False,
            'n_jobs': 1,
            'penalize_length': True
        }
    }
    if path and os.path.isfile(path):
        try:
            with open(path, 'r') as f:
                user_cfg = yaml.load(f, Loader=yaml.FullLoader)
            # merge
            for k,v in user_cfg.items():
                default[k] = v
        except Exception:
            pass
    # Wrap to attribute-style access
    cfg = SimpleNamespace(**{k:v for k,v in default.items() if k!='Inference'})
    cfg.Inference = SimpleNamespace(**default['Inference'])
    return cfg

class SR4MDLRegressor:
    """轻量封装，提供与原 SymQRegressor 相似接口。
    若模型或权重不可用，退化为简单的多项式拟合 (degree=2)。
    """
    def __init__(self):
        self.cfg = None
        self.model = None  # MDLformer
        self.est = None    # MCTS4MDL
        self.sympy_expr = None
        self._fallback = False
        self._vars = []

    def setup_model(self):
        if Tokenizer is None or MCTS4MDL is None or MDLformer is None:
            print("sr4mdl core modules not available, using polynomial fallback")
            self._fallback = True
            return
        weights_path = getattr(self.cfg.Inference, 'weights_path', 'weights/checkpoint.pth')
        weights_path = Path(__file__).parent / weights_path
        if not weights_path.exists():
            print(f"sr4mdl weights not found: {weights_path}, fallback to polynomial fit")
            self._fallback = True
            return
        device = 'cpu'
        if getattr(self.cfg.Inference, 'use_gpu', False) and torch.cuda.is_available():
            device = 'cuda'
        try:
            state_dict = torch.load(str(weights_path), map_location=device)
            args = AttrDict(
                dropout=0.1,
                d_model=512,
                d_input=64,
                d_output=512,
                n_TE_layers=8,
                max_len=50,
                max_param=5,
                max_var=max(10, getattr(self.cfg, 'num_vars', 3)),
                uniform_sample_number=self.cfg.Inference.n_points,
                device=device,
                use_SENet=True,
                use_old_model=False
            )
            if device == 'auto' and AutoGPU:
                args.device = AutoGPU().choice_gpu(memory_MB=1486, interval=15)
            tokenizer = Tokenizer(-100, 100, 4, args.max_var)
            self.model = MDLformer(args, state_dict['xy_token_list'])
            self.model.load(state_dict['xy_encoder'], state_dict['xy_token_list'], strict=True)
            self.model.eval()
            self.est = MCTS4MDL(
                tokenizer=tokenizer,
                model=self.model,
                n_iter= min(2000, getattr(self.cfg, 'max_iter', 10000)),
                keep_vars=True,
                normalize_y=False,
                normalize_all=False,
                remove_abnormal=True,
                binary=['Mul','Div','Add','Sub'],
                unary=['Sqrt','Cos','Sin','Pow2','Pow3','Exp','Log','Inv','Neg'],
                leaf=[1,2,np.pi],
                log_per_sec=120,
            )
        except Exception as e:
            print(f"Failed to initialize sr4mdl model: {e}. Fallback to polynomial fit.")
            self._fallback = True

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._vars = [f'x{i+1}' for i in range(X.shape[1])]
        if self._fallback or self.est is None:
            # simple polynomial (degree 2) regression using lstsq
            feats = [np.ones(len(X))]
            feats += [X[:, i] for i in range(X.shape[1])]
            feats += [X[:, i]**2 for i in range(X.shape[1])]
            A = np.vstack(feats).T
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            expr_terms = [f"{coef[0]:.4g}"]
            idx = 1
            for i,v in enumerate(self._vars):
                expr_terms.append(f"({coef[idx]:.4g})*{v}"); idx+=1
            for i,v in enumerate(self._vars):
                expr_terms.append(f"({coef[idx]:.4g})*{v}**2"); idx+=1
            self.sympy_expr = ' + '.join(expr_terms)
            self._coef = coef
            return
        # Prepare dict input for estimator
        X_dict = {v: X[:, i] for i,v in enumerate(self._vars)}
        try:
            self.est.fit(X_dict, y, use_tqdm=False)
            self.sympy_expr = str(self.est.eqtree)
        except Exception as e:
            print(f"sr4mdl search failed ({e}); fallback to polynomial fit")
            self._fallback = True
            self.fit(X, y)

    def predict(self, X: np.ndarray):
        if self._fallback or self.est is None:
            feats = [np.ones(len(X))]
            feats += [X[:, i] for i in range(X.shape[1])]
            feats += [X[:, i]**2 for i in range(X.shape[1])]
            A = np.vstack(feats).T
            return A @ getattr(self, '_coef', np.zeros(A.shape[1]))
        X_dict = {v: X[:, i] for i,v in enumerate(self._vars)}
        return self.est.predict(X_dict)

def complexity(est):
    try:
        return len(est.eqtree)
    except Exception:
        return 0

def model(est, X=None):
    try:
        return est.eqtree.to_str()
    except Exception:
        return 'unknown'

if __name__ == '__main__':
    # 简单自测
    cfg = load_cfg('')
    reg = SR4MDLRegressor()
    reg.cfg = cfg
    reg.setup_model()
    X = np.random.randn(50,2)
    y = 2*X[:,0] - 0.5*X[:,1] + 0.3
    reg.fit(X,y)
    print('Expr:', reg.sympy_expr)
    print('Pred shape', reg.predict(X).shape)
