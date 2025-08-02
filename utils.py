import re
import sympy as sp
import numpy as np
from sympy.parsing.sympy_parser import parse_expr
from collections import OrderedDict

SCREEN_HEIGHT = 750
SCREEN_WIDTH = 1200

# Used for parsing problem formulation
def parse_mathematical_expression(expr_str):
    expr_str = expr_str.strip()
    
    if not expr_str: 
        return None
    
    expr_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr_str)
    expr_str = re.sub(r'([a-zA-Z])\(', r'\1*(', expr_str)
    expr_str = re.sub(r'\)([a-zA-Z])', r')*\1', expr_str)
    
    try:
        # equation handling
        if '=' in expr_str and ('<' not in expr_str and '>' not in expr_str):
            left, right = expr_str.split('=', 1)
            left_expr = sp.parse_expr(left.strip())
            right_expr = sp.parse_expr(right.strip())
        elif '<=' in expr_str:
            left, right = expr_str.split('<=', 1)
            left_expr = sp.parse_expr(left.strip())
            right_expr = sp.parse_expr(right.strip())
            return sp.Le(left_expr, right_expr)
        elif '>=' in expr_str:
            left, right = expr_str.split('>=', 1)
            left_expr = sp.parse_expr(left.strip())
            right_expr = sp.parse_expr(right.strip())
            return sp.Ge(left_expr, right_expr)
        elif '<' in expr_str and '=' not in expr_str:
            left, right = expr_str.split('<', 1)
            left_expr = sp.parse_expr(left.strip())
            right_expr = sp.parse_expr(right.strip())
            return sp.Lt(left_expr, right_expr)
        elif '>' in expr_str and '=' not in expr_str:
            left, right = expr_str.split('>', 1)
            left_expr = sp.parse_expr(left.strip())
            right_expr = sp.parse_expr(right.strip())
            return sp.Gt(left_expr, right_expr)
        else:
            return sp.parse_expr(expr_str)
    except Exception as e:
        print(f"Error parsing expression: {e}")
        return None
    
# print(parse_mathematical_expression("2x+4y<=3"))
# c = [parse_mathematical_expression("2x+4y").coeff(x) for x in list(parse_mathematical_expression("2x+4y").free_symbols)]
# print(c)

def convert_constraints_to_matrix(constraints: list):
    """
    Converts a list of SymPy inequality expressions into matrix form A * x <= b.
    
    Args:
        constraints (list): List of SymPy inequality expressions (e.g., [2*x + 3*y <= 4])
    
    Returns:
        tuple: (A_ub, b_ub) where A_ub is a list of lists, and b_ub is a list of RHS values.
    """
    A_ub = []
    b_ub = []

    for con in constraints:
        lhs = con.lhs
        rhs = con.rhs

        # Convert >= to <= by multiplying both sides by -1
        if con.rel_op == ">=":
            lhs = -lhs
            rhs = -rhs

        row = [float(lhs.coeff(v)) for v in lhs.free_symbols if lhs]
        A_ub.append(row)
        b_ub.append(float(rhs))

    return np.array(A_ub), np.array(b_ub)

def convert_objective_to_vector(obj_func):
    """
    Converts a SymPy objective function into a vector form.
    
    Args:
        objective (sympy.Expr): The objective function expression.
    
    Returns:
        list: Coefficients of the objective function corresponding to the variables.
    """
    return np.array( [float(obj_func.coeff(var)) for var in list(obj_func.free_symbols) if obj_func] )


def sorted_variables(expr: sp.Expr):
    """
    Sorts SymPy symbols based on their first appearance in a SymPy expression.

    Args:
        expr (sp.Expr): A SymPy expression (e.g., 2*x + 3*y + z).
        var_list (list[sp.Symbol], optional): Subset of variables to sort. 
            If None, uses expr.free_symbols.

    Returns:
        list[sp.Symbol]: Sorted list of symbols in order of first appearance.
    """
    return sorted(expr.free_symbols, key = lambda symbol: symbol.name)

print(convert_objective_to_vector(parse_mathematical_expression("2profit+3x87")))
print(parse_mathematical_expression("2profit+3x87").free_symbols)

expr = "2profit+3x88+8x9+9x12+9profit2"
vars = []
for i in parse_mathematical_expression(expr).free_symbols:
    if str(i) in expr:
        vars.append(i)
free_syms = sorted(parse_mathematical_expression(expr).free_symbols, key = lambda symbol: symbol.name)
print(free_syms)

ordered_symbols = {}
counter = 1

# Traverse the expression tree nodes
for item in sp.preorder_traversal(parse_mathematical_expression(expr)):
    # Check if the item is a symbol and has not been seen before
    if isinstance(item, sp.Symbol) and item.name not in ordered_symbols:
        ordered_symbols[item.name] = counter
        counter += 1
        
print(ordered_symbols)


# print(convert_constraints_to_matrix([sp.sympify('2*x + 3*y <= 4'), sp.sympify('x + 2*y <= 1')]))
# expr = parse_expr("2x3+3x2")
# print(sorted_variables(expr))
    return [float(objective.coeff(v)) for v in get_variables(objective)]


from scipy.optimize import linprog
import numpy as np

def solve_lp(A: np.ndarray, b: np.ndarray, c: np.ndarray, sense: str ='max'):
    """
    Solves the LP:
        Maximize:    cᵀx
        Subject to:  Ax ≤ b
                     x ≥ 0

    Parameters:
        A (ndarray): Constraint matrix (m x n)
        b (ndarray): RHS vector (length m)
        c (ndarray): Coefficients of objective function (length n)

    Returns:
        x_opt (ndarray): Optimal solution vector
        z_opt (float):   Optimal objective value
    """
    # Since linprog minimizes, we negate c to turn max into min
    if sense == 'min':
        res = linprog(c, A_ub=A, b_ub=b, method='highs')
    elif sense == 'max':
        res = linprog(-c, A_ub=A, b_ub=b, method='highs')
    else:
        raise ValueError("Sense must be either 'min' or 'max'")

    if res.success:
        x_opt = res.x
        z_opt = c @ x_opt  # Compute max value (undo negation)
        return x_opt, z_opt
    if res.success:
        x_opt = res.x
        z_opt = c @ x_opt  # Undo negation
        return x_opt, z_opt
    else:
        # Distinguish between unbounded and infeasible
        if res.status == 3:
            raise ValueError("The problem is unbounded: objective function can increase indefinitely.")
        elif res.status == 2:
            raise ValueError("The problem is infeasible: no solution satisfies all constraints.")
        else:
            raise ValueError(f"LP solve failed: {res.message}")

A = np.array([
    [1, 1],
    [2, 7]
])
b = np.array([1, 6])  # Now conflicting
c = np.array([1, 2])  # Objective function coefficients

x_opt, z_opt = solve_lp(A, b, c)

print("Optimal solution:", x_opt)
print("Optimal value:", z_opt)

