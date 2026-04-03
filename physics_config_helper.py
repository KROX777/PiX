"""
Physics problem configuration helper.

Helper for configuring physics discovery problems with the following structure:
1. Problem name and description
2. Variable names
3. Constant values (name=value format)
4. Derived quantities and their definitions (use ? for unknown definitions requiring hypothesis enumeration)
5. Known physics equations
6. Decision tree entries, each row contains:
   - Hypothesis ID
   - Hypothesis name
   - Parent node ID
   - Requires symbolic regression (true/false)
   - Optional: Expression (eq: <expression>)
   - Optional: Definition (def: [new_var_1; new_var_2; ...] var=...)
   - Optional: Constraints (constraint: [var]...>0 / [var]...>=0)
   - Optional: Related variables (related: var1; var2; var3; var4)
7. Symbolic regression function library and nesting configuration
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
        # 1. Problem name and description
        name, description = lines[0].split('|', 1)
        # 2. Variable names
        variables = [v.strip() for v in lines[1].split('|')]
        # 3. Constants (format: variable_name=value, separated by |)
        constants = {}
        if lines[2] == '(void)':
            constants = {}
        else:
            for const in lines[2].split('|'):
                if '=' in const:
                    var, value = const.split('=', 1)
                    constants[var.strip()] = float(value.strip())
        # 4. Derived quantities and their definitions (format: variable_name=definition, separated by |)
        derived_quantities = {}
        unknown_variables = []
        if len(lines) > 3 and lines[3].strip():
            for dq in lines[3].split('|'):
                if '=' in dq:
                    var, expr = dq.split('=', 1)
                    if expr.strip() == '?':
                        unknown_variables.append(var.strip())
                    derived_quantities[var.strip()] = expr.strip()
        # 5. Known physics equations
        known_equations = []
        i = 4
        while i < len(lines) and lines[i].strip():
            if lines[i] == 'end':
                i += 1
                break
            known_equations.append(lines[i].strip())
            i += 1
        # 6. Decision tree entries
        decision_tree_rows = []
        while i < len(lines) and lines[i].strip():
            if lines[i] == 'end':
                i += 1
                break
            row = [x.strip() for x in lines[i].split('|')]
            if len(row) > 1:
                decision_tree_rows.append(row)
            i += 1
        # 7. Allow nesting configuration (single line: yes/no or true/false)
        allow_nesting = False
        if i < len(lines):
            allow_nesting = 'yes' in lines[i].lower() or 'true' in lines[i].lower()
            i += 1
        # 8. Symbolic regression function library (unary and binary functions, currently only unary supported)
        unary_functions = []
        binary_functions = []
        if i < len(lines):
            unary_functions = [f.strip() for f in lines[i].split('|') if f.strip()]
            i += 1
        # if i < len(lines):
        #     binary_functions = [f.strip() for f in lines[i].split('|') if f.strip()]
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
        # Convert hypotheses node
        hypotheses = []
        for row in self.problem.decision_tree_rows:
            # Hypothesis ID, name, parent node ID, requires symbolic regression,
            # optional equation, optional definitions, optional constraints, optional related variables
            node = {
                'id': int(row[0]) if len(row) > 0 else None,  # 1-based index
                'name': row[1] if len(row) > 1 else None,
                'father_node': int(row[2]) if len(row) > 2 else None,  # ID of father node
                'require_sr': row[3].lower() in ['true', '1', 'yes', 'y'] if len(row) > 3 else False,
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
            
        # Organize YAML structure
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
        from anytree import Node, RenderTree

        root = Node("Root")
        nodes = {0: root}
        
        for row in self.problem.decision_tree_rows:
            node_id = int(row[0])
            node_name = row[1] if len(row) > 1 else f"Node {node_id}"
            nodes[node_id] = Node(node_name)
        for row in self.problem.decision_tree_rows:
            node_id = int(row[0])
            parent_id = int(row[2])
            nodes[node_id].parent = nodes[parent_id]
            
        for pre, _, node in RenderTree(root):
            print(f"{pre}{node.name}")

if __name__ == "__main__":
    config = PhysicsConfig()
    config.from_csv('physics_problem.csv')
    os.makedirs('pix/cfg/problem', exist_ok=True)
    config.to_yaml(f'pix/cfg/problem/{config.problem.name}.yaml')
    config.visualize_tree()