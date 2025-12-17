# hippylib 反问题场参数估计深度学习笔记

## 目录
1. [库概述](#1-库概述)
2. [核心数学理论](#2-核心数学理论)
3. [核心模块详解](#3-核心模块详解)
4. [反问题求解流程](#4-反问题求解流程)
5. [关键算法实现](#5-关键算法实现)
6. [实践应用指南](#6-实践应用指南)

---

## 1. 库概述

### 1.1 hippylib 简介
**hippylib** (Inverse Problem PYthon library) 是专门用于 PDE 约束的确定性和贝叶斯反问题的库。

**核心特性**：
- 🔧 基于 **FEniCS** 进行 PDE 离散化（有限元方法）
- ⚡ 基于 **PETSc** 进行并行线性代数运算
- 📈 维度独立的算法（成本不随参数维度增加）
- 🎯 自动微分计算梯度和 Hessian
- 🔍 低秩 Hessian 近似用于后验协方差

**适用问题**：
- 参数识别（coefficient field inversion）
- 不确定性量化（uncertainty quantification）
- 贝叶斯反演（Bayesian inference）
- 数据同化（data assimilation）

---

## 2. 核心数学理论

### 2.1 反问题的数学描述

#### 正问题（Forward Problem）
给定参数 $m$，求解状态 $u$：
$$
\begin{cases}
-\nabla \cdot (a(m) \nabla u) = f & \text{in } \Omega \\
u = u_0 & \text{on } \partial\Omega
\end{cases}
$$

常见形式：$a(m) = \exp(m)$ 或 $a(m) = m$

#### 反问题（Inverse Problem）
给定观测数据 $u_d$，恢复参数 $m$：
$$
\min_{m} J(m) = \underbrace{\frac{1}{2}\|u(m) - u_d\|^2_{W}}_{\text{misfit}} + \underbrace{\frac{\gamma}{2}\|\nabla m\|^2}_{\text{regularization}}
$$

其中：
- $u(m)$：正问题的解
- $u_d$：观测数据（含噪声）
- $\gamma$：正则化参数
- $W$：误差权重矩阵

### 2.2 拉格朗日方法与伴随方法

#### 拉格朗日函数
$$
\mathcal{L}(u,m,p) = \frac{1}{2}(u-u_d, u-u_d) + \frac{\gamma}{2}(\nabla m, \nabla m) + (\exp(m)\nabla u, \nabla p) - (f, p)
$$

其中 $p$ 是拉格朗日乘子（伴随变量）。

#### 最优性条件（KKT条件）
1. **状态方程（正问题）**：
   $$\mathcal{L}_p = 0 \Rightarrow (\exp(m)\nabla u, \nabla \tilde{p}) = (f, \tilde{p}), \quad \forall \tilde{p}$$

2. **伴随方程**：
   $$\mathcal{L}_u = 0 \Rightarrow (\exp(m)\nabla p, \nabla \tilde{u}) = -(u-u_d, \tilde{u}), \quad \forall \tilde{u}$$

3. **梯度方程**：
   $$\mathcal{L}_m = 0 \Rightarrow \gamma(\nabla m, \nabla \tilde{m}) + (\tilde{m}\exp(m)\nabla u, \nabla p) = 0, \quad \forall \tilde{m}$$

#### 梯度的显式表达
$$
\nabla_m J(m) = \gamma \nabla^2 m + \exp(m)(\nabla u \cdot \nabla p)
$$

**关键优势**：只需求解一次正问题和一次伴随问题，即可得到完整梯度（与参数维度无关）。

### 2.3 Hessian 矩阵的结构

#### 完整 Hessian（Newton 方法）
$$
\mathcal{H}(m)(\hat{m}) = \gamma \nabla^2 \hat{m} + C^T W_{uu} C + C^T W_{um} + W_{mu} C + W_{mm}
$$

其中：
- $C = \frac{\partial u}{\partial m}$：参数-状态的灵敏度矩阵
- $W_{uu}$：状态-状态二阶导数
- $W_{um}, W_{mu}$：混合二阶导数
- $W_{mm}$：参数-参数二阶导数

#### Gauss-Newton 近似
忽略二阶项 $W_{um}, W_{mu}, W_{mm}$：
$$
\mathcal{H}_{GN}(m)(\hat{m}) = \gamma \nabla^2 \hat{m} + C^T W_{uu} C
$$

**何时使用 Gauss-Newton**：
- 前几次迭代（远离最优点）
- 噪声水平较高
- 计算成本要求较低

### 2.4 Hessian-向量乘积（无矩阵方法）

计算 $H\hat{m}$ 需要求解两个线性系统：

1. **增量正问题**：
   $$(\exp(m)\nabla \hat{u}, \nabla \tilde{p}) = -(\hat{m}\exp(m)\nabla u, \nabla \tilde{p})$$

2. **增量伴随问题**：
   $$(\exp(m)\nabla \hat{p}, \nabla \tilde{u}) = -(W_{uu}\hat{u}, \tilde{u}) - (W_{um}\hat{m}, \tilde{u})$$

3. **组装结果**：
   $$H\hat{m} = \gamma R\hat{m} + C^T\hat{p} + W_{mu}\hat{u} + W_{mm}\hat{m}$$

---

## 3. 核心模块详解

### 3.1 模块架构

```
hippylib/
├── modeling/               # 建模核心
│   ├── model.py           # Model 类：整合所有组件
│   ├── prior.py           # 先验分布（LaplacianPrior, BiLaplacianPrior）
│   ├── misfit.py          # 目标函数（DiscreteStateObservation）
│   ├── PDEProblem.py      # PDE 问题抽象基类
│   ├── PDEVariationalProblem.py  # 变分形式的 PDE
│   ├── reducedHessian.py  # Reduced Hessian 算子
│   └── posterior.py       # 后验分布与不确定性量化
│
├── algorithms/            # 优化算法
│   ├── NewtonCG.py        # Inexact Newton-CG 方法
│   ├── cgsolverSteihaug.py  # Steihaug-CG 求解器
│   ├── randomizedEigensolver.py  # 低秩特征值求解
│   └── linalg.py          # 线性代数工具
│
├── mcmc/                  # MCMC 采样
├── utils/                 # 工具函数
│   ├── random.py          # 并行随机数生成
│   └── finite_diff.py     # 有限差分工具
└── forward_uq/            # 前向不确定性量化
```

### 3.2 Model 类详解

`Model` 是 hippylib 的核心类，整合了所有组件：

```python
class Model:
    def __init__(self, problem, prior, misfit):
        self.problem = problem  # PDEProblem: 定义 PDE
        self.prior = prior      # Prior: 先验分布
        self.misfit = misfit    # Misfit: 目标函数
        self.gauss_newton_approx = False
```

**主要方法**：

1. **generate_vector(component)**: 生成适当形状的向量
   ```python
   x = model.generate_vector()  # 返回 [u, m, p]
   m = model.generate_vector(PARAMETER)
   ```

2. **solveFwd(out, x)**: 求解正问题
   - 输入：`x = [u_init, m, p]`（仅使用 m 和 u_init）
   - 输出：`out`（求解的状态 u）

3. **solveAdj(out, x)**: 求解伴随问题
   - 输入：`x = [u, m, p]`（使用 u 和 m）
   - 输出：`out`（求解的伴随 p）

4. **evalGradientParameter(x, mg)**: 计算梯度
   - 返回：梯度范数

5. **setPointForHessianEvaluations(x, gauss_newton_approx)**: 设置 Hessian 评估点

6. **solveFwdIncremental(sol, rhs)**: 求解增量正问题

7. **solveAdjIncremental(sol, rhs)**: 求解增量伴随问题

### 3.3 Prior 类详解

#### LaplacianPrior

协方差算子：$C = (\delta I - \gamma \Delta)^{-1}$

```python
prior = LaplacianPrior(
    Vh,                    # 有限元空间
    gamma=0.1,            # 控制相关长度
    delta=0.5,            # 控制方差
    mean=None,            # 先验均值
    rel_tol=1e-12,        # 求解器容差
    solver_type="krylov"  # 或 "lu"
)
```

**关键参数**：
- $\gamma$：越大，相关长度越长（场越光滑）
- $\delta$：越大，方差越小（场越接近均值）
- 比值 $\gamma/\delta$ 控制相关长度尺度

**主要方法**：
```python
# 采样
noise = dl.Vector()
prior.init_vector(noise, "noise")
parRandom.normal(1., noise)
m_sample = dl.Vector()
prior.init_vector(m_sample, 0)
prior.sample(noise, m_sample, add_mean=True)

# 计算先验成本
cost = prior.cost(m)  # 0.5 * (m - m_mean)^T R (m - m_mean)

# 计算梯度
grad = dl.Vector()
prior.init_vector(grad, 0)
prior.grad(m, grad)  # grad = R * (m - m_mean)
```

#### BiLaplacianPrior

协方差算子：$C = (\delta I - \gamma \Delta)^{-2}$（更光滑）

```python
prior = BiLaplacianPrior(
    Vh, gamma, delta,
    anis_diff=None,     # 各向异性扩散张量
    robin_bc=True       # Robin 边界条件
)
```

### 3.4 Misfit 类详解

#### DiscreteStateObservation

用于点观测或离散观测：

```python
# 构建观测算子 B
B = assemblePointwiseObservation(Vh, obs_points)

# 创建 misfit
misfit = DiscreteStateObservation(
    B,                    # 观测算子
    data=d_obs,          # 观测数据
    noise_variance=0.01  # 噪声方差
)
```

**主要方法**：
```python
# 计算 misfit cost
cost = misfit.cost(x)  # 0.5/noise_var * ||B*u - d||^2

# 计算梯度
grad_u = model.generate_vector(STATE)
misfit.grad(STATE, x, grad_u)  # B^T * (B*u - d) / noise_var

grad_m = model.generate_vector(PARAMETER)
misfit.grad(PARAMETER, x, grad_m)  # 通常为 0

# 设置线性化点
misfit.setLinearizationPoint(x, gauss_newton_approx=False)

# 应用 Hessian 块
out = model.generate_vector(STATE)
misfit.apply_ij(STATE, STATE, dir, out)  # W_uu * dir
```

### 3.5 ReducedHessian 类详解

无矩阵实现的 Hessian 算子：

```python
class ReducedHessian:
    def __init__(self, model, misfit_only=False):
        self.model = model
        self.misfit_only = misfit_only
        self.ncalls = 0  # 记录调用次数
    
    def mult(self, x, y):
        """应用 Hessian: y = H * x"""
        if self.gauss_newton_approx:
            self.GNHessian(x, y)
        else:
            self.TrueHessian(x, y)
        self.ncalls += 1
    
    def GNHessian(self, x, y):
        """Gauss-Newton 近似"""
        # 1. 增量正问题：C * x
        self.model.applyC(x, self.rhs_fwd)
        self.model.solveFwdIncremental(self.uhat, self.rhs_fwd)
        
        # 2. 增量伴随问题：W_uu * uhat
        self.model.applyWuu(self.uhat, self.rhs_adj)
        self.model.solveAdjIncremental(self.phat, self.rhs_adj)
        
        # 3. C^T * phat
        self.model.applyCt(self.phat, y)
        
        # 4. 加上正则化项
        if not self.misfit_only:
            self.model.applyR(x, self.yhelp)
            y.axpy(1., self.yhelp)
    
    def TrueHessian(self, x, y):
        """完整 Newton Hessian"""
        # 类似 GNHessian，但包含二阶项
        # W_um, W_mu, W_mm
```

---

## 4. 反问题求解流程

### 4.1 完整工作流程

```python
# ===== 步骤 1: 设置网格和函数空间 =====
mesh = dl.UnitSquareMesh(64, 64)
Vh_state = dl.FunctionSpace(mesh, 'Lagrange', 2)  # 状态：二阶
Vh_param = dl.FunctionSpace(mesh, 'Lagrange', 1)  # 参数：一阶
Vh = [Vh_state, Vh_param, Vh_state]

# ===== 步骤 2: 定义正问题 =====
def u_boundary(x, on_boundary):
    return on_boundary

bc = dl.DirichletBC(Vh[STATE], dl.Constant(0.0), u_boundary)
bc0 = dl.DirichletBC(Vh[STATE], dl.Constant(0.0), u_boundary)

f = dl.Constant(1.0)

def pde_varf(u, m, p):
    return ufl.exp(m) * ufl.inner(ufl.grad(u), ufl.grad(p)) * ufl.dx - f * p * ufl.dx

pde = PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)

# ===== 步骤 3: 定义先验 =====
gamma = 0.1
delta = 0.5
prior = BiLaplacianPrior(Vh[PARAMETER], gamma, delta)

# 生成真实参数
noise = dl.Vector()
prior.init_vector(noise, "noise")
parRandom.normal(1., noise)
mtrue = dl.Vector()
prior.init_vector(mtrue, 0)
prior.sample(noise, mtrue)

# ===== 步骤 4: 生成合成观测数据 =====
utrue = pde.generate_state()
x = [utrue, mtrue, None]
pde.solveFwd(utrue, x)

# 构建观测算子
ntargets = 100
np.random.seed(1)
targets = np.random.uniform(0.1, 0.9, [ntargets, 2])
B = assemblePointwiseObservation(Vh[STATE], targets)

# 添加噪声
rel_noise = 0.01
utrue_obs = dl.Vector()
B.init_vector(utrue_obs, 0)
B.mult(utrue, utrue_obs)
noise_level = rel_noise * utrue_obs.norm("linf")
noise_vec = dl.Vector()
B.init_vector(noise_vec, 0)
parRandom.normal(noise_level, noise_vec)
data = utrue_obs + noise_vec

# ===== 步骤 5: 定义 misfit =====
noise_variance = noise_level**2
misfit = DiscreteStateObservation(B, data, noise_variance)

# ===== 步骤 6: 构建 Model =====
model = Model(pde, prior, misfit)

# ===== 步骤 7: 设置初始猜测 =====
m0 = dl.interpolate(dl.Constant(0.0), Vh[PARAMETER])
x = model.generate_vector()
x[STATE].zero()
x[PARAMETER].axpy(1., m0.vector())

# ===== 步骤 8: 优化求解 =====
parameters = ReducedSpaceNewtonCG_ParameterList()
parameters["rel_tolerance"] = 1e-6
parameters["abs_tolerance"] = 1e-9
parameters["max_iter"] = 20
parameters["GN_iter"] = 5  # 前5次用 Gauss-Newton
parameters["globalization"] = "LS"  # 线搜索
parameters["cg_coarse_tolerance"] = 0.5

solver = ReducedSpaceNewtonCG(model, parameters)

# 求解
x = solver.solve(x)

print("Converged:", solver.converged)
print("Reason:", ReducedSpaceNewtonCG.termination_reasons[solver.reason])
print("Iterations:", solver.it)
print("Total CG iterations:", solver.total_cg_iter)
print("Final gradient norm:", solver.final_grad_norm)

# ===== 步骤 9: 可视化结果 =====
m_MAP = x[PARAMETER]
u_MAP = x[STATE]

# 绘图
plt.figure(figsize=(15, 5))
plt.subplot(131)
dl.plot(dl.Function(Vh[PARAMETER], mtrue))
plt.title("True Parameter")
plt.subplot(132)
dl.plot(dl.Function(Vh[PARAMETER], m_MAP))
plt.title("MAP Estimate")
plt.subplot(133)
dl.plot(dl.Function(Vh[PARAMETER], mtrue - m_MAP))
plt.title("Error")
plt.show()
```

### 4.2 hippytest.py 的实现细节

你的代码手动实现了核心算法，让我们对照理解：

```python
# 你的实现                          # hippylib 等价
# ============================================================

# 1. 梯度计算
CT_p = dl.Vector()
C.init_vector(CT_p, 1)
C.transpmult(p.vector(), CT_p)      # C^T * p
MG = CT_p + R * m.vector()          # C^T*p + R*m（正则化项）
dl.solve(M, g, MG)                  # g = M^{-1} * MG（预条件）

# hippylib 中：
gradnorm = model.evalGradientParameter(x, mg)

# 2. Hessian 应用（Gauss-Newton）
# 增量正问题
rhs = -(self.C * v)
bc_adj.apply(rhs)
dl.solve(self.A, self.du, rhs)      # 求解 A * du = -C * v

# 增量伴随问题
rhs = -(self.W * self.du)
bc_adj.apply(rhs)
dl.solve(self.adj_A, self.dp, rhs)  # 求解 A_adj * dp = -W * du

# 组装 Hessian
self.R.mult(v, y)                   # 正则化项
self.C.transpmult(self.dp, self.CT_dp)
y.axpy(1, self.CT_dp)               # 加上 C^T * dp

# hippylib 中：
HessApply = ReducedHessian(model)
HessApply.mult(mhat, result)

# 3. Newton-CG 优化
solver = CGSolverSteihaug()
solver.set_operator(Hess_Apply)
solver.set_preconditioner(Psolver)
solver.solve(m_delta, -MG)          # 求解 H * m_delta = -g

# hippylib 中：
solver = ReducedSpaceNewtonCG(model, parameters)
x = solver.solve(x)

# 4. 线搜索
while descent == 0 and no_backtrack < 10:
    m.vector().axpy(alpha, m_delta)
    # 求解正问题
    # 检查 Armijo 条件
    if cost_new < cost_old + alpha * c * MG.inner(m_delta):
        descent = 1
    else:
        alpha *= 0.5

# hippylib 自动处理
```

---

## 5. 关键算法实现

### 5.1 Inexact Newton-CG 算法

```
算法：Inexact Newton-CG with Line Search

输入：初始猜测 m_0, 容差 tol, 最大迭代次数 max_iter
输出：最优参数 m*

for k = 0, 1, 2, ... until convergence:
    1. 求解正问题：
       给定 m_k，求解 u_k
    
    2. 求解伴随问题：
       给定 u_k, m_k，求解 p_k
    
    3. 计算梯度：
       g_k = ∇_m J(m_k) = γR*m_k + C^T*p_k
       
       if ||g_k|| < tol:
           收敛，退出
    
    4. 求解 Newton 系统（用 CG）：
       H_k * Δm_k = -g_k
       
       其中 H_k 是 Hessian（或 Gauss-Newton 近似）
       
       CG 容差：tol_cg = min(0.5, sqrt(||g_k||/||g_0||))
       （Eisenstat-Walker 准则）
    
    5. 线搜索（Armijo 规则）：
       α = 1
       while J(m_k + α*Δm_k) > J(m_k) + c*α*(g_k, Δm_k):
           α = α/2
       
       m_{k+1} = m_k + α*Δm_k
    
    6. 检查收敛：
       if ||g_{k+1}|| < tol or |(g_k, Δm_k)| < tol_gdm:
           收敛，退出
```

**关键技巧**：

1. **Gauss-Newton 转 Newton**：
   - 前几次迭代（如前5次）：使用 GN 近似（更快，更稳定）
   - 后续迭代：使用完整 Newton（二次收敛）

2. **Eisenstat-Walker CG 容差**：
   ```python
   tolcg = min(0.5, sqrt(gradnorm / gradnorm_ini))
   ```
   自适应调整 CG 精度，避免过度求解

3. **Steihaug 规则**：
   CG 遇到负曲率时，停在信赖域边界

### 5.2 伴随方法计算梯度

**算法步骤**：

```python
def compute_gradient(m):
    """
    输入：参数 m
    输出：梯度 ∇J(m)
    """
    # 1. 正问题：求解 u(m)
    solve_forward(u, m)  # F(u, m, p) = 0 for all p
    
    # 2. 伴随问题：求解 p(u, m)
    #    右端项：∂J/∂u = W * (u - u_d)
    rhs = W * (u - u_d)
    solve_adjoint(p, u, m, rhs)  # ∂F/∂u^T * p = -rhs
    
    # 3. 组装梯度
    grad = γ * R * m + C^T * p
    #    = 正则化梯度 + misfit 梯度
    
    return grad
```

**为什么这么做？**

直接有限差分计算梯度：
$$\frac{\partial J}{\partial m_i} \approx \frac{J(m + h e_i) - J(m)}{h}$$
需要 $n$ 次正问题求解（$n$ = 参数维度）

伴随方法：
- 1 次正问题 + 1 次伴随问题 = **维度独立**！

### 5.3 低秩后验协方差近似

后验协方差：
$$\Gamma_{\text{post}} = (H_{\text{misfit}} + \Gamma_{\text{prior}}^{-1})^{-1}$$

**问题**：$H$ 是 $n \times n$ 密集矩阵（$n \sim 10^4$-$10^6$），无法显式存储和求逆！

**解决方案**：低秩近似

1. **广义特征值问题**：
   $$H_{\text{misfit}} V = \Gamma_{\text{prior}}^{-1} V \Lambda$$

2. **只保留前 $r$ 个大特征值**：
   $$H_{\text{misfit}} \approx \Gamma_{\text{prior}}^{-1} V_r \Lambda_r V_r^T \Gamma_{\text{prior}}^{-1}$$

3. **Sherman-Morrison-Woodbury 公式**：
   $$\Gamma_{\text{post}} \approx \Gamma_{\text{prior}} - V_r D_r V_r^T$$
   
   其中 $D_r = \text{diag}(\lambda_i/(\lambda_i+1))$

**实现**：

```python
from hippylib import ReducedHessian, doublePassG

# 1. 构建 Hessian 算子
Hmisfit = ReducedHessian(model, misfit_only=True)

# 2. 随机特征值求解
r = 50  # 保留特征值数量
Omega = MultiVector(model.generate_vector(PARAMETER), r+10)
parRandom.normal(1., Omega)

d, V = doublePassG(
    Hmisfit,                    # Hessian 算子
    prior.R,                    # 先验精度矩阵
    prior.Rsolver,              # 先验协方差矩阵
    Omega,                      # 随机向量
    r
)

# d: 特征值（从大到小）
# V: 特征向量

# 3. 从后验采样
prior_sample = dl.Vector()
prior.init_vector(prior_sample, 0)
prior.sample(noise, prior_sample)

# 低秩校正
post_sample = prior_sample.copy()
for i in range(r):
    correction = d[i]/(d[i]+1)
    Vip = V[i].inner(prior.R * prior_sample)
    post_sample.axpy(-correction * Vip, V[i])
```

---

## 6. 实践应用指南

### 6.1 参数选择指南

#### 先验参数
```python
# LaplacianPrior: C = (δI - γΔ)^{-1}
gamma = 0.1   # ↑ 增加 → 相关长度 ↑（更光滑）
delta = 0.5   # ↑ 增加 → 方差 ↓（更确定）

# 相关长度尺度约为 sqrt(γ/δ)
correlation_length = np.sqrt(gamma / delta)
```

**选择建议**：
- 地下水流：`gamma=0.1, delta=1.0`（中等光滑）
- 热传导：`gamma=0.01, delta=0.1`（很光滑）
- 地震学：`gamma=1.0, delta=10.0`（较粗糙）

#### 优化参数
```python
parameters["rel_tolerance"] = 1e-6    # 相对梯度容差
parameters["abs_tolerance"] = 1e-9    # 绝对梯度容差
parameters["max_iter"] = 20           # 最大迭代次数
parameters["GN_iter"] = 5             # Gauss-Newton 迭代次数
parameters["cg_coarse_tolerance"] = 0.5  # CG 最粗容差
parameters["cg_max_iter"] = 100       # CG 最大迭代次数
```

**调参技巧**：
1. 如果不收敛，增加 `max_iter`
2. 如果 CG 迭代过多，降低 `cg_coarse_tolerance`
3. 如果早期震荡，增加 `GN_iter`
4. 噪声大时，用 `GN_iter = max_iter`（全用 GN）

### 6.2 常见问题与解决

#### 问题 1：正问题求解失败
```
RuntimeError: Newton solver did not converge
```

**原因**：参数更新步长太大，导致正问题非线性求解失败

**解决**：
```python
# 方法 1：减小线搜索步长
parameters["LS"]["c_armijo"] = 1e-3  # 默认 1e-4

# 方法 2：使用信赖域
parameters["globalization"] = "TR"

# 方法 3：使用更保守的初始猜测
m0 = prior.mean  # 用先验均值初始化
```

#### 问题 2：梯度不下降
```
Iteration 5: ||grad|| = 1.234e-3 (not decreasing)
```

**原因**：
- Hessian 不正定
- CG 求解器精度不够
- 参数空间病态

**解决**：
```python
# 增强正定性
parameters["GN_iter"] = max_iter  # 始终用 GN

# 提高 CG 精度
parameters["cg_coarse_tolerance"] = 0.1

# 增加正则化
gamma *= 10  # 增强先验强度
```

#### 问题 3：计算太慢
```
每次迭代需要 10 分钟...
```

**加速技巧**：
```python
# 1. 使用更粗的网格
mesh = dl.UnitSquareMesh(32, 32)  # 而非 128x128

# 2. 降低 CG 迭代次数
parameters["cg_max_iter"] = 50

# 3. 使用 LU 分解（小规模问题）
prior = LaplacianPrior(Vh, gamma, delta, solver_type="lu")

# 4. 并行运行（需要 MPI）
mpirun -n 4 python your_script.py
```

### 6.3 验证与调试

#### 梯度检验
```python
from hippylib import modelVerify

# 有限差分验证梯度
h = 1e-6
err = modelVerify(model, x, h, 1)  # 1 = PARAMETER
print("Gradient error:", err)
# 应该 < 1e-4
```

#### Hessian 检验
```python
# 有限差分验证 Hessian
err_H = modelVerify(model, x, h, 2)  # 2 = Hessian
print("Hessian error:", err_H)
# 应该 < 1e-3
```

#### 收敛曲线
```python
# 记录每次迭代
costs = []
grads = []

def callback(it, x):
    cost = model.cost(x)[0]
    grad = model.generate_vector(PARAMETER)
    gradnorm = model.evalGradientParameter(x, grad)
    costs.append(cost)
    grads.append(gradnorm)
    print(f"Iteration {it}: cost={cost:.3e}, ||grad||={gradnorm:.3e}")

solver = ReducedSpaceNewtonCG(model, parameters, callback=callback)
x = solver.solve(x)

# 绘制
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.semilogy(costs)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.subplot(122)
plt.semilogy(grads)
plt.xlabel("Iteration")
plt.ylabel("||Gradient||")
plt.show()
```

### 6.4 扩展到自己的问题

#### 模板：自定义 PDE 问题

```python
# 1. 定义你的 PDE
def my_pde_varf(u, m, p):
    """
    定义你的 PDE 弱形式
    
    例如：非线性扩散
    -∇·(m^2 ∇u) = f
    """
    return m**2 * ufl.inner(ufl.grad(u), ufl.grad(p)) * ufl.dx - f * p * ufl.dx

# 2. 设置边界条件
def boundary_func(x, on_boundary):
    return on_boundary and x[0] < dl.DOLFIN_EPS  # 左边界

bc = dl.DirichletBC(Vh[STATE], u_boundary_value, boundary_func)

# 3. 创建 PDE 问题
pde = PDEVariationalProblem(
    Vh, my_pde_varf, bc, bc0,
    is_fwd_linear=False  # 非线性！
)

# 4. 其余步骤相同
prior = BiLaplacianPrior(Vh[PARAMETER], gamma, delta)
misfit = DiscreteStateObservation(B, data, noise_variance)
model = Model(pde, prior, misfit)
# ... 求解
```

#### 模板：自定义观测算子

```python
class MyCustomObservation(Misfit):
    """自定义观测"""
    def __init__(self, targets, data, noise_var):
        self.targets = targets  # 观测位置
        self.data = data        # 观测数据
        self.noise_var = noise_var
        
        # 构建观测矩阵 B
        self.B = self._build_observation_operator()
        self.Bu = dl.Vector()
        self.B.init_vector(self.Bu, 0)
    
    def _build_observation_operator(self):
        # 实现你的观测算子
        # 例如：积分观测、边界通量等
        pass
    
    def cost(self, x):
        self.B.mult(x[STATE], self.Bu)
        diff = self.Bu - self.data
        return 0.5 / self.noise_var * diff.inner(diff)
    
    def grad(self, i, x, out):
        if i == STATE:
            self.B.mult(x[STATE], self.Bu)
            diff = self.Bu - self.data
            self.B.transpmult(diff, out)
            out *= 1.0 / self.noise_var
        else:
            out.zero()
    
    def setLinearizationPoint(self, x, gauss_newton_approx=False):
        pass
    
    def apply_ij(self, i, j, dir, out):
        if i == STATE and j == STATE:
            self.B.mult(dir, self.Bu)
            self.B.transpmult(self.Bu, out)
            out *= 1.0 / self.noise_var
        else:
            out.zero()
```

---

## 7. 高级主题

### 7.1 贝叶斯反演

从确定性反演到贝叶斯：

**确定性**：寻找单一最优参数
$$\hat{m} = \arg\min J(m)$$

**贝叶斯**：计算参数的后验分布
$$\pi(m | d) \propto \pi(d | m) \pi(m)$$

**Laplace 近似**：
$$\pi(m | d) \approx \mathcal{N}(m_{MAP}, \Gamma_{post})$$

其中：
- $m_{MAP}$：最大后验估计（同确定性解）
- $\Gamma_{post}$：后验协方差

**实现**：
```python
# 1. 求 MAP 点
solver = ReducedSpaceNewtonCG(model)
x = solver.solve(x)
m_MAP = x[PARAMETER]

# 2. 构建后验协方差
from hippylib import Posterior, LowRankHessian

posterior = Posterior(model)
posterior.setLinearizationPoint(x)

# 3. 计算低秩 Hessian
Hmisfit = LowRankHessian(posterior, r=50)

# 4. 后验采样
samples = [dl.Vector() for _ in range(100)]
for s in samples:
    posterior.init_vector(s, 0)
    posterior.sample(noise, s)
```

### 7.2 时间依赖问题

```python
from hippylib import TimeDependentPDEVariationalProblem

# 定义时间依赖的 PDE
def pde_varf_tv(u, m, p):
    u_t, u_x = u  # 时间导数，空间解
    return (
        u_t * p * ufl.dx +
        ufl.exp(m) * ufl.inner(ufl.grad(u_x), ufl.grad(p)) * ufl.dx
    )

# 设置时间离散
T = 1.0
dt = 0.01
pde = TimeDependentPDEVariationalProblem(
    Vh, pde_varf_tv, bc, bc0,
    T=T, dt=dt, theta=0.5  # Crank-Nicolson
)
```

### 7.3 并行计算

hippylib 自动支持 MPI 并行：

```bash
# 单机多核
mpirun -n 8 python inverse_problem.py

# 集群
srun -n 256 python inverse_problem.py
```

代码无需修改！FEniCS 和 PETSc 会自动处理：
- 网格分区
- 矩阵/向量分布
- 并行线性求解器

---

## 8. 总结与最佳实践

### 核心要点

1. **伴随方法是关键**
   - 维度独立的梯度计算
   - 一次正问题 + 一次伴随问题

2. **Hessian 不需要显式构造**
   - 通过增量问题实现 Hessian-向量乘积
   - Gauss-Newton 近似加速初期迭代

3. **低秩近似降维**
   - 后验协方差的低秩表示
   - 只需要几十个特征向量

4. **先验很重要**
   - 控制解的光滑性
   - 平衡数据拟合和正则化

### 工作流程检查清单

- [ ] 网格和函数空间设置正确
- [ ] PDE 弱形式验证（手动求解一次）
- [ ] 边界条件正确施加
- [ ] 观测数据质量检查（噪声水平）
- [ ] 先验参数合理（correlation length）
- [ ] 初始猜测不会导致正问题失败
- [ ] 梯度验证通过（有限差分检查）
- [ ] Hessian 验证通过（对小问题）
- [ ] 优化参数调优（GN_iter, tolerance）
- [ ] 收敛曲线合理（单调下降）
- [ ] 结果可视化和物理检验

### 常用代码片段

```python
# 快速设置标准反问题
def setup_standard_inverse_problem(mesh, gamma, delta, obs_points, data, noise_var):
    # 函数空间
    Vh_state = dl.FunctionSpace(mesh, 'Lagrange', 2)
    Vh_param = dl.FunctionSpace(mesh, 'Lagrange', 1)
    Vh = [Vh_state, Vh_param, Vh_state]
    
    # PDE
    bc = dl.DirichletBC(Vh[STATE], dl.Constant(0.0), "on_boundary")
    def pde_varf(u, m, p):
        return ufl.exp(m) * ufl.inner(ufl.grad(u), ufl.grad(p)) * ufl.dx
    pde = PDEVariationalProblem(Vh, pde_varf, bc, bc, is_fwd_linear=True)
    
    # 先验
    prior = BiLaplacianPrior(Vh[PARAMETER], gamma, delta)
    
    # Misfit
    B = assemblePointwiseObservation(Vh[STATE], obs_points)
    misfit = DiscreteStateObservation(B, data, noise_var)
    
    # Model
    model = Model(pde, prior, misfit)
    
    return model, Vh

# 使用
model, Vh = setup_standard_inverse_problem(
    mesh, gamma=0.1, delta=0.5,
    obs_points=targets, data=observations,
    noise_var=0.01
)
```

---

## 参考资源

1. **hippylib 文档**: https://hippylib.readthedocs.io
2. **教程**: `../hippylib/tutorial/`
   - `2_PoissonDeterministic.ipynb`: 确定性反演
   - `3_SubsurfaceBayesian.ipynb`: 贝叶斯反演
   - `4_AdvectionDiffusionBayesian.ipynb`: 时间依赖
3. **论文**:
   - Villa et al., "hIPPYlib: An Extensible Software Framework for Large-Scale Inverse Problems", JOSS 2018
4. **FEniCS 文档**: https://fenicsproject.org
5. **你的代码**: `hippytest.py` - 手动实现的优秀参考！

---

## 下一步计划

将 hippylib 应用到你的 PiX 项目：

1. **参数识别**：从 PDE 数据反推物理参数
2. **方程发现**：结合符号回归，识别 PDE 形式
3. **不确定性量化**：量化识别参数的不确定性
4. **数据同化**：融合观测数据改进 PDE 模型

这些都可以利用 hippylib 的强大功能！
