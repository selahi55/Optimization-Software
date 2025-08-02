import re
import sympy as sp

SCREEN_HEIGHT = 750
SCREEN_WIDTH = 1200

# Used for parsing problem formulation
def parse_mathematical_expression(expr_str):
    """
    Parse a mathematical expression string like '3x+2y' into a SymPy expression.
    
    Args:
        expr_str (str): Mathematical expression string
        
    Returns:
        sympy.Expr: Parsed SymPy expression
    """
    expr_str = expr_str.strip()
    
    # Handle implicit multiplication (3x -> 3*x, 2y -> 2*y)
    expr_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr_str)
    expr_str = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', expr_str)
    expr_str = re.sub(r'([a-zA-Z])([a-zA-Z])', r'\1*\2', expr_str)
    
    # Handle cases like x(y) -> x*y
    expr_str = re.sub(r'([a-zA-Z])\(', r'\1*(', expr_str)
    expr_str = re.sub(r'\)([a-zA-Z])', r')*\1', expr_str)
    
    try:
        # parsing using SymPy
        return sp.parse_expr(expr_str)
         
    except Exception as e:
        print(f"Error parsing expression: {e}")
        return None

def get_variables(expr):
    """Extract all variables from a SymPy expression."""
    if expr is None:
        return []
    return list(expr.free_symbols)


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

        row = [float(lhs.coeff(v)) for v in get_variables(lhs)]
        A_ub.append(row)
        b_ub.append(float(rhs))

    return A_ub, b_ub

def convert_objective_to_vector(objective: list):
    """
    Converts a SymPy objective function into a vector form.
    
    Args:
        objective (sympy.Expr): The objective function expression.
    
    Returns:
        list: Coefficients of the objective function corresponding to the variables.
    """
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

