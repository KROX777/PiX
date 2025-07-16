"""
物理问题配置助手
1. 问题名字，简介
2. 变量名
3. 常量输入
4. 导出物理量及定义式（或标注定义式未知、需要枚举假设，标注方法为=?) (请保证到此为止所有变量名都通过变量、常量或定义式方式给出)
5. 物理定律预输入
6. 决策树输入，每一行 (假设编号，假设名称, 父节点, 是否需要符号回归, 表达式 (eq: <表达式>), 定义式 (def: [new_var_1; new_var_2; ...] var=...), 限制 (constraint: [var]...>0 / [var]...>=0），某变量与某变量有关（related: var1; var2; var3; var4）
7. 符号回归的函数库，是否允许嵌套
"""
import csv
import os
import yaml
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class SymbolicRegressionConfig:
    function_expressions: List[str]
    allow_nesting: bool
    
    def to_dict(self):
        return {
            'functions': self.function_expressions,
            'allow_nesting': self.allow_nesting
        }

@dataclass
class PhysicsProblem:
    name: str
    description: str
    variables: List[str]
    derived_quantities: Dict[str, str]
    known_equations: List[str]
    constants: Dict[str, float]
    decision_tree_rows: List[List[str]]
    unknown_variables: List[str]
    symbolic_regression_config: SymbolicRegressionConfig

class PhysicsConfig:
    def __init__(self):
        self.problem: Optional[PhysicsProblem] = None

    def from_csv(self, csv_file: str):
        with open(csv_file, 'r', encoding='utf-8') as f:
            lines = [line.rstrip('\n') for line in f if line.strip()]
        # 1. 问题名字,简介
        name, description = lines[0].split(',', 1)
        # 2. 变量名
        variables = [v.strip() for v in lines[1].split(',')]
        # 3. 常量输入（一个变量名=一个值，逗号分隔）
        constants = {}
        if lines[2] == '(void)':
            constants = {}
        else:
            for const in lines[2].split(','):
                if '=' in const:
                    var, value = const.split('=', 1)
                    constants[var.strip()] = float(value.strip())
        # 4. 导出物理量及其定义式（一个变量名=一个定义式，逗号分隔）
        derived_quantities = {}
        unknown_variables = []
        if len(lines) > 3 and lines[3].strip():
            for dq in lines[3].split(','):
                if '=' in dq:
                    var, expr = dq.split('=', 1)
                    # 把expr中的;换成,
                    expr = expr.replace(';', ',')
                    if expr.strip() == '?':
                        unknown_variables.append(var.strip())
                    derived_quantities[var.strip()] = expr.strip()
        # 5. 物理定律
        known_equations = []
        i = 4
        while i < len(lines) and lines[i].strip():
            if lines[i] == 'end':
                i += 1
                break
            known_equations.append(lines[i].strip())
            i += 1
        # 6. 决策树输入
        decision_tree_rows = []
        while i < len(lines) and lines[i].strip():
            if lines[i] == 'end':
                i += 1
                break
            row = [x.strip() for x in lines[i].split(',')]
            if len(row) > 1:
                decision_tree_rows.append(row)
            i += 1
        # 7. 是否允许嵌套（单独一行）
        allow_nesting = False
        if i < len(lines):
            allow_nesting = '是' in lines[i] or 'true' in lines[i].lower()
            i += 1
        # 8. 符号回归函数库（最后两行：一元函数和二元函数，现仅支持一元）
        unary_functions = []
        binary_functions = []
        if i < len(lines):
            unary_functions = [f.strip() for f in lines[i].split(',') if f.strip()]
            i += 1
        # if i < len(lines):
        #     binary_functions = [f.strip() for f in lines[i].split(',') if f.strip()]
        #     i += 1
        sr_config = SymbolicRegressionConfig(function_expressions=unary_functions, allow_nesting=allow_nesting)
        self.problem = PhysicsProblem(
            name=name.strip(),
            description=description.strip(),
            variables=variables,
            derived_quantities=derived_quantities,
            known_equations=known_equations,
            decision_tree_rows=decision_tree_rows,
            symbolic_regression_config=sr_config,
            constants=constants,
            unknown_variables=unknown_variables
        )
        
    def to_yaml(self, output_file: str):
        import yaml
        if self.problem is None:
            raise ValueError("Problem is not initialized. Call from_csv() first.")
        # hypotheses节点转换
        hypotheses = []
        for row in self.problem.decision_tree_rows:
            # 假设编号，假设名称, 父节点, 是否需要符号回归, 方程表达式（可选），使得哪个变量与哪个变量有关
            node = {
                'id': int(row[0]) if len(row) > 0 else None, # 1-based
                'name': row[1] if len(row) > 1 else None,
                'father_node': int(row[2]) if len(row) > 2 else None, # id of father node
                'require_sr': row[3].lower() in ['true', '1', 'yes', 'y', '是'] if len(row) > 3 else False,
                'equation': [],
                'definitions': [],
                'constraints': [],
                'related_variables': []
            }
            for i in range(4, len(row)):
                if row[i].startswith('eq:'):
                    node['equation'].append(row[i][3:].strip())
                elif row[i].startswith('def:'):
                    node['definitions'].append(row[i][4:].strip())
                elif row[i].startswith('constraint:'):
                    node['constraints'].append(row[i][11:].strip())
                elif row[i].startswith('related:'):
                    vars = row[i][8:].strip().split(';')
                    node['related_variables'].append(vars)
            hypotheses.append(node)
            
        # 组织yaml结构
        problem_dict = {
            'name': self.problem.name,
            'description': self.problem.description,
            'variables': self.problem.variables,
            'constants': self.problem.constants,
            'derived_quantities': self.problem.derived_quantities,
            'known_equations': self.problem.known_equations,
            'hypotheses': hypotheses,
            'symbolic_regression_config': self.problem.symbolic_regression_config.to_dict(),
            'unknown_variables': self.problem.unknown_variables
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.dump(problem_dict, f, allow_unicode=True, default_flow_style=False)
    
    def visualize_tree(self):
        """可视化决策树"""
        from anytree import Node, RenderTree
        if self.problem is None:
            raise ValueError("Problem is not initialized. Call from_csv() first.")
        
        # 创建根节点
        root = Node(self.problem.name)
        nodes = {self.problem.name: root}
        
        # 创建所有节点
        for row in self.problem.decision_tree_rows:
            node_name = row[1] if len(row) > 1 else "Unnamed"
            parent_name = row[2] if len(row) > 2 else self.problem.name
            node = Node(node_name, parent=nodes[parent_name])
            nodes[node_name] = node
        
        # 打印树结构
        for pre, _, node in RenderTree(root):
            print(f"{pre}{node.name}")

if __name__ == "__main__":
    config = PhysicsConfig()
    config.from_csv('physics_problem.csv')
    os.makedirs('pix/cfg/problem', exist_ok=True)
    config.to_yaml(f'pix/cfg/problem/{config.problem.name}.yaml')