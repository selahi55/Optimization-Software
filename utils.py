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


print(convert_constraints_to_matrix([sp.sympify('2*x + 3*y <= 4'), sp.sympify('x + y >= 1')]))