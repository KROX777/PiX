# HVAE-SNIP 集成两种版本的实现规划

本文件只描述设计，不写具体代码，方便后续照着实现和检查。

---

## 总体约束与目标

- 以 `symbolicregression/envs/generators.py` 中的生成器（`RandomFunctions`）为**唯一真源**：运算符集合、变量命名和连续系数的取值空间都以它为准。
- 系数：
  - 由 `generate_float` 和相关逻辑决定，取值空间是**有限离散集合**（mantissa + 有界 exponent），总 token 数量约 \(10^4\) 级。
  - **要求：每一个实际出现过的浮点系数都对应一个单独的 token，HVAE 必须能精确恢复这些 token，不使用统一的 `CONSTANT` 占位符。**
- 两种版本的差异：
  - **版本 1：HVAE 的输入输出格式完全等于 generator 的表达式格式**（运算符名、变量名、系数 token 都一致，树结构也以 generator 为基准）；
  - **版本 2：HVAE 维持原始 EDHiE 格式（SRToolkit / SymbolLibrary + 自己的 Node / tokens_to_tree），generator 通过一层前处理 / 后处理适配进去。**

以下只详细规划 **版本 1** 的实现方法；版本 2 会单独在代码里实现。

---

## 版本 1：HVAE 直接在 generator 表达式上工作

### 1. 统一的数据结构与接口

目标：让 HVAE 不再依赖 SRToolkit 的 `Node` 和 `SymbolLibrary`，而是直接使用 generator 的 `Node` 和一套本地的 `SnipSymbolLibrary`，但保持 HVAE 的网络结构（`BatchedNode / Encoder / Decoder / GRU221 / GRU122`）不变。

**关键思想**：
- HVAE 的 Encoder/Decoder 只关心两件事：
  1. 一个“批量树”的结构，即 `BatchedNode`：每个位置对应一个 token index（one-hot / target），以及左右子树；
  2. 一个符号库：能够把符号字符串 ↔ index 映射，并且知道某个符号是函数(`fn`)还是叶子(`var`/`const`/`lit`)。
- generator 的 `Node` 已经给出了树结构和符号字符串；我们只需要：
  - 定义一个 `SnipSymbolLibrary`：
    - 内部维护 `List[str] symbols`，其中每个 symbol 是 **generator 中实际出现过的字符串**，包括：
      - 所有运算符：`operators_real` + `operators_extra`；
      - 所有变量名：`x_0, x_1, ...`（上限从 generator 参数 `max_input_dimension` 推出）；
      - 所有离散系数 token：从训练数据扫描得到的有限集合；
      - 可能还要包括 `rand`, `e`, `pi`, `euler_gamma` 等；
    - 提供方法：
      - `symbols2index() -> Dict[str, int]`
      - `get_type(symbol: str) -> Literal["fn", "var", "const", "lit"]`
        - `symbol in all_operators` → `"fn"`
        - `symbol.startswith("x_")` → `"var"`
        - 其余（数值常量、`e`, `pi` 等）→ `"const"` 或 `"lit"`（只要和 Decoder 里叶子判断一致即可）。
  - 实现一对：`Node ↔ BatchedNode` 的转换（而不是先转 token 再 tokens_to_tree）。

### 2. 设计 `SnipSymbolLibrary`

文件建议：`symbolicregression/HVAE/snip_symbol_library.py`

**接口草图（不写具体代码）：**

- 类 `SnipSymbolLibrary`：
  - 初始化参数：
    - `operators: Iterable[str]`：来自 `all_operators` 的 key（`add, sub, mul, ...`）。
    - `variables: Iterable[str]`：例如 `[f"x_{i}" for i in range(max_input_dimension)]`。
    - `constants: Iterable[str]`：根据训练集中出现的所有常数字符串构建（见下一小节）。
  - 内部：
    - `self.symbols: List[str]`：按固定顺序拼接：
      - 先运算符，再变量，再常数，以保证 index 稳定；
    - `self._sym2idx: Dict[str, int]`：从 symbols 构造。
  - 方法：
    - `symbols2index(self) -> Dict[str, int]`：返回 `self._sym2idx`；
    - `index2symbols(self) -> Dict[int, str]`：反向映射（Decoder.sample_symbol 用得到）。
    - `get_type(self, s: str) -> str`：
      - `s in all_operators` → `'fn'`
      - `s.startswith('x_')` → `'var'`
      - 否则 → `'const'` 或 `'lit'`。

> 注：这里的 `get_type` 只需要与 HVAE Decoder 的两个逻辑兼容即可：
> - "是函数" → 会继续展开左右子树；
> - "是叶子" → 不再向下展开。

### 3. 精确系数 token 的获取策略

要求：**每个实际出现过的浮点数都是一个独立的 token**；不枚举整个理论空间，只用“训练数据中实际出现的那个有限子集”。

步骤：

1. 用 `RandomFunctions` 生成训练用的表达式树：
   - 利用 `generate_multi_dimensional_tree`，固定 `output_dimension=1` 或小值；
   - 对每个 `Node`，调用 `tree.prefix()` 或自写 DFS，扫描所有 `value` 是数值的节点；
2. 判断是否为数值的规则与 generator 里 `function_to_skeleton` 一致：
   - 尝试 `float(pre)` 成功且不是纯整数（或根据需求包含整数）→ 认为是系数；
   - `math_constants`（`e`, `pi`, `euler_gamma`）单独加进常数集合；
3. 把所有出现过的数值字符串加入一个 `set`，最终变成 `List[str]` 排序后作为 `constants` 传给 `SnipSymbolLibrary`。

这样：
- 训练时 HVAE 的词表是 **“训练数据中所有真正出现过的符号 + 系数”**；
- 以后再扩充训练集时，可以重新扫描得到更大的常数集合，更新词表并重训；
- 对于 stable diffusion 等应用，**离散 token 空间是完全精确地覆盖了训练表达式中的所有系数**。

### 4. `Node` → `BatchedNode` 的构造（训练方向）

目标：绕开 SRToolkit 的 `tokens_to_tree`，直接从 generator 的 `Node` 构造 HVAE 所需的 `BatchedNode` 树。

回顾 HVAE 的结构：
- `BatchedNode.symbols`: 长度 = batch_size，一批树在同一个“位置”上的符号；
- `BatchedNode.left/right`: 递归地指向左右孩子的 `BatchedNode`；
- `BatchedNode.create_target()`：
  - 为每个 batch 位置构造 one-hot target 矩阵和 mask；
- `Encoder.recursive_forward(tree: BatchedNode)`：
  - 对每个 `BatchedNode` 调用 `self.gru(tree.target, h_left, h_right)`，其中 `tree.target` 是 shape = (batch_size, vocab_size) 的 one-hot；
- `Decoder.recursive_forward`：
  - 也是在 `BatchedNode` 树上递归，把 prediction 存到每个节点上；

所以：

1. 对于单棵 generator `Node`，我们需要先把它变成**二叉树结构**，因为 HVAE 的 `BatchedNode` 是二叉树：
   - generator 的 `Node.children` 是 N 叉的（可 >2）；
   - 映射策略：
     - 对于一元运算符（`abs`, `sin`, `log`, `pow2`, ...）：保持为单子树（直接放在 `left`）；
     - 对于多元运算符（`add`, `mul` 等），我们在版本 1 里采用**左结合的二叉展开**：
       - 例如：`add(a, b, c, d)` →`(((a + b) + c) + d)`，对应的二叉树：
         - 根：`add(left=add(left=add(left=a,right=b), right=c), right=d)`。
       - 对于 generator 中本来就是二元的运算（大部分 `operators_real`），保持不变。
   - 这一步可以定义一个函数：`to_binary_tree(node: Node) -> Node`（返回仍然是 generator 的 `Node`，只是 children 被规约成二叉）。

2. 构造批量树 `BatchedNode` 的策略：
   - HVAE 的构造方式是：
     - 初始时，`BatchedNode(symbol2index, size=<batch_size>)`，`symbols` 是长度为 batch_size 的列表；
     - 每加入一棵树，调用 `add_tree(tree)`，会在当前层加一个元素，并递归扩展到子树；
   - 对于我们的版本 1，可以沿用原始 `BatchedNode` 的逻辑，不改源码：
     - 对每一批训练表达式（列表 `[Node, Node, ...]`），我们直接：
       - `batched = BatchedNode(symbol2index, trees=trees_batch)`。
     - 其中 `trees_batch` 已经是经过 `to_binary_tree` 转换的 generator `Node`。
   - 唯一需要对齐的是：generator 的 `Node` 属性名要符合 `BatchedNode.add_tree` 里访问的接口：
     - 现在 `BatchedNode.add_tree` 假定 `tree.symbol`, `tree.left`, `tree.right`，而 generator 的 `Node` 是 `value` + `children`；
     - 解决方法：
       - 实现一个适配器类 `SnipBinaryNode`：
         - 包装 generator 的 `Node`，暴露 `symbol/left/right` 三个属性：
           - `symbol` = 原 `value`；
           - `left/right` 指向包装后的左右子树；
       - 或者写一个简单的“转换函数”，把 generator 的二叉 `Node` 复制成一个本地 `Node` 类（结构和 SRToolkit 的 Node 相同：`symbol, left, right`）。

3. 训练循环：
   - 不使用 EDHiE 的 `tokens_to_tree` 和 `generate_n_expressions`；
   - 封装一个新的 `TreeDataset`（可以重用 EDHiE 的 `TreeDataset` 类，里面只是 list 包装），内部存的是我们自己的 `Node` 类型；
   - `create_batch`（批量构造）可以重用 EDHiE 的实现，只要我们传入的是前面提到的适配后的 Node；
   - 其余 `HVAE` / `train_hvae` 几乎不需要修改，唯一需要的是：
     - 初始化时不要从 SRToolkit 的 `SymbolLibrary` 导入，而是用我们的 `SnipSymbolLibrary` 实例。

### 5. 解码与系数恢复（推理方向）

Encoder/Decoder 不做特殊处理，系数恢复的关键点在于：

1. Decoder 在 `decode(z)` 时：
   - 通过 `sample_symbol` 从 `index2symbol` 中选出具体字符串 token，比如：`"add"`, `"x_0"`, `"0.03125"`；
   - 类型判断全由 `SnipSymbolLibrary.get_type` 决定：
     - 如果是 `fn`，则继续展开左右子树；
     - 如果是 `var/const/lit`，则停止展开，这个节点就是叶子；
2. 得到的 `BatchedNode` → 表达式树：
   - 通过 `BatchedNode.to_expr_list()` 拿到一批树（这一步可以复用 EDHiE 的实现，只要 Node 类型兼容）；
   - 或者自己写一个 `batched_to_snip_nodes`，把 `symbols` 数组还原为 generator 的 `Node` 树：
     - 利用我们在训练方向构造二叉树时使用的同样规则反推；
3. 常数 token 已经是**具体数值字符串**：
   - 直接在 SNIP 端用 `float(token)` 恢复成数值；
   - 组合出完整的表达式树 `Node`，可以用现有的 `val(x)` 等方法做数值评估。

这样，版本 1 满足：
- 符号空间完全等于 generator 的输出空间（包括实数系数）；
- HVAE 不再引入任何新的占位符；
- 保持 HVAE 原有的 Encoder/Decoder 结构，只替换了“外壳”：符号库和树类型的实现。

### 6. 与版本 2 的关系

- 版本 2 使用 SRToolkit / 原始 EDHiE 的 token + Node 表示，再通过前后处理映射回 generator；
- 版本 1 则完全绕开 SRToolkit 的表达式层，直接在 generator 表达式上训练和解码；
- 这两套实现可以共存：
  - `EDHiE` 目录下保留原作者的 pipeline（需要 SRToolkit 和数据集）；
  - `snip_symbol_library.py` + 一个新的 `train_hvae_snip_v1.py` 则专门用于版本 1；
  - 版本 2 的桥接脚本（例如 `train_snip_hvae_v2.py`）则调用原有的 `train_hvae`，只是前后多了一层表达式格式转换。

> 后续真正实现版本 1 的时候，建议：
> - 先只支持一小部分运算符（`add, sub, mul, div, sin, cos, exp, log, pow2, pow3` 等），写好单元测试验证 Node ↔ BatchedNode ↔ Node 的往返不丢信息；
> - 再逐步把 generator 中其它运算符和特殊符号（`rand` 等）加入 `SnipSymbolLibrary` 和类型判断中。
