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


print(get_variables(parse_mathematical_expression("2x+3y")))