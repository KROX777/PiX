from pix.calculator import Calculator
import sympy as sp
import copy

class TreeNode:
    def __init__(self, id, name=None, father_node=None, equation=None, constraints=None, related_variables=None, definitions=None):
        self.id = id
        self.name = name
        self.father_node = father_node
        self.equation = equation if equation is not None else []
        self.constraints = constraints if constraints is not None else []
        self.related_variables = related_variables if related_variables is not None else []
        self.definitions = definitions if definitions is not None else []
        self.activated = False
        self.children_nodes = []  # List of child node IDs

    def __repr__(self):
        return f"TreeNode(name={self.name}, father_node={self.father_node}, equation={self.equation}, constraints={self.constraints}, related_variables={self.related_variables})"
    
class LightTree:
    """
    A lightweight hypotheses tree structure not containing calculator.
    """
    def __init__(self, config, root_dir):
        self.root = TreeNode(id=0, name="root")
        self.nodes = [self.root]
        for hyp in config.problem['hypotheses']:
            self.add_node(hyp['id'], name=hyp['name'], father_node=hyp['father_node'], 
                         equation=hyp.get('equation', []), constraints=hyp.get('constraints', []),
                         related_variables=hyp.get('related_variables', []),
                         definitions=hyp.get('definitions', []))
    
    def add_node(self, id, name=None, father_node=None, equation=None, constraints=None, related_variables=None, definitions=None):
        new_node = TreeNode(id, name=name, father_node=father_node, equation=equation, 
                            constraints=constraints, related_variables=related_variables,
                            definitions=definitions)
        self.nodes.append(new_node)
        self.nodes[father_node].children_nodes.append(id)  # Add to father's children
        return new_node
    
    def generate_all_possibilities(self):
        """
        Returns:
            List[List[int]]
        """
        if not self.root.children_nodes:
            return [[]]

        all_subtree_paths = []
        for root_child in self.root.children_nodes:
            subtree_paths = self._get_all_paths_from_node(root_child)
            all_subtree_paths.append(subtree_paths)
        
        possibilities = []
        self._cartesian_product(all_subtree_paths, [], 0, possibilities)
        
        return possibilities
    
    def gen_all_paths_from_deci_list(self, deci_list):
        """
        Returns:
            List[List[int]]
        """
        all_subtree_paths = []
        for deci in deci_list:
            subtree_paths = self._get_all_paths_from_node(deci)
            all_subtree_paths.append(subtree_paths)
        
        possibilities = []
        self._cartesian_product(all_subtree_paths, [], 0, possibilities)
        
        return possibilities

    def _get_all_paths_from_node(self, node_id):
        """
        Returns:
            List[List[int]]
        """
        children = self.nodes[node_id].children_nodes
        if not children:
            return [[node_id]]
        
        all_paths = []
        for child_id in children:
            child_paths = self._get_all_paths_from_node(child_id)
            for path in child_paths:
                all_paths.append([node_id] + path)
        
        return all_paths

    def _cartesian_product(self, all_subtree_paths, current_combination, index, result):
        """
        计算所有子树路径的笛卡尔积
        
        Args:
            all_subtree_paths: 每个根子节点的所有可能路径
            current_combination: 当前正在构建的组合
            index: 当前处理的根子节点索引
            result: 存储所有结果的列表
        """
        if index == len(all_subtree_paths):
            flattened = []
            for path in current_combination:
                flattened.extend(path)
            result.append(flattened)
            return

        for path in all_subtree_paths[index]:
            self._cartesian_product(all_subtree_paths, current_combination + [path], index + 1, result)

    def get_tree_structure(self):
        """
        辅助方法：打印树结构以便调试
        """
        print("Tree structure:")
        for node in self.nodes:
            if node.id == 0:
                print(f"Root (ID: {node.id})")
            else:
                indent = "  " * self._get_depth(node.id)
                print(f"{indent}├─ {node.name} (ID: {node.id}, Father: {node.father_node})")

    def _get_depth(self, node_id):
        """获取节点深度"""
        depth = 0
        current_id = node_id
        while current_id != 0:
            for node in self.nodes:
                if node.id == current_id:
                    current_id = node.father_node
                    depth += 1
                    break
        return depth
    

class HypothesesTree(LightTree):
    def __init__(self, config, root_dir): # virtual root node
        super().__init__(config, root_dir)
        self.calculator = Calculator(config, root_dir)
        self.original_calculator = copy.deepcopy(self.calculator)
        
    def reset(self):
        self.calculator = copy.deepcopy(self.original_calculator)

    def activate_node(self, node_id, verbose=False):
        """
        Returns:
        3 Lists of sympy expr: residuals of new hypo, constraints of new hypo (for simple regression), and related variables for sr.
        """

        if self.nodes[node_id].activated:
            print(f"Node {node_id} is already activated.")
            return [], [], []
        self.nodes[node_id].activated = True
        
        new_hypo = self.nodes[node_id]
        # if verbose == True:
        #     print("Activating Hypo: ", new_hypo)
        
        # Register new variables from definitions; SR not here
        for defi in new_hypo.definitions:
            if defi.startswith('[') and ']' in defi:
                vars_str = defi[1:defi.index(']')].strip()
                defi = defi[defi.index(']') + 1:].strip()
                
                # Parse variable names separated by semicolons or spaces
                if vars_str:
                    var_names = [v.strip() for v in vars_str.replace(';', ' ').split() if v.strip()]
                    for var in var_names:
                        self.calculator.register_unknown_var(var)
                        
            if '=' in defi:
                var, expr = defi.split('=', 1)
                var = var.strip()
                expr = expr.strip()
                self.calculator.update_unknown_var(var, expr)
            else:
                raise ValueError(f"Invalid definition format: {defi}")
        
        for eq in new_hypo.equation:
            if '=' in eq:
                left, right = eq.split('=', 1)
                eq_str = f"({left}) - ({right})"
                self.calculator.get_new_equation(eq_str)
            else:
                self.calculator.get_new_equation(eq)
        
        for constraint_str in new_hypo.constraints:
            # Parse constraint format: [var]expression>threshold
            original_constraint = constraint_str
            threshold = "0"
            
            # Extract threshold
            if '>=' in constraint_str:
                constraint_str, threshold = constraint_str.split('>=', 1)
            elif '>' in constraint_str:
                constraint_str, threshold = constraint_str.split('>', 1)
            
            # Extract variable name and expression
            if constraint_str.startswith('[') and ']' in constraint_str:
                var = constraint_str[1:constraint_str.index(']')].strip()
                expr_str = constraint_str[constraint_str.index(']') + 1:].strip()
                self.calculator.add_constraint(expr_str, var)
            else:
                print(f"Warning: Constraint '{original_constraint}' does not have proper format [var]expression>threshold")
        
        related_vars = []
        self.calculator.upd_local_dict()
        for rel in new_hypo.related_variables:
            if len(rel) >= 2:
                # Format: [y_var, x_var1, x_var2, ...]
                y = rel[0]
                x_vars = rel[1:] if len(rel) > 1 else []
                flag = True
                if y not in self.calculator.sp_unknown_quantities:
                    print(f"Warning: Variable '{y}' already known or unregistered, can't SR.")
                    flag = False
                for x in x_vars:
                    if x in self.calculator.sp_unknown_quantities:
                        print(f"Warning: Variable '{x}' unknown, can't SR.")
                        flag = False
                    if x not in self.calculator.local_dict:
                        print(f"Warning: Variable '{x}' not in local_dict, can't SR.")
                        flag = False
                if flag == False:
                    continue
                related_vars.append(tuple((y, x_vars)))
            else:
                print(f"Warning: {rel} must have at least 2 variables for SR.")
        
        return related_vars