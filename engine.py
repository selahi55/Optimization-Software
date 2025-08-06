# Engine which solves the formulation provided by the user
import numpy as np
import sympy as sp
import scipy as scp
from utils import *

class Engine:
    def __init__(self, sense, obj_func, constraints):
        # problem formulation details
        self.sense = sense
        self.obj_func = obj_func
        self.constraints = constraints

    def simplex_scipy(self):
        # Extract coefficients for objective and constraints
        parsed_obj_func = parse_mathematical_expression(self.obj_func)
        variables = list(parsed_obj_func.free_symbols)
        c = np.array([float(parsed_obj_func.coeff(var)) for var in variables])
        A, b = convert_constraints_to_matrix( [parse_mathematical_expression(constraint) for constraint in self.constraints] )
        bounds = [(0, None) for _ in variables]  # Non-negative variables

        # Determine sense
        if self.sense.lower() == 'maximization':
            c = -c  

        result = scp.optimize.linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        return result

    # Simplex Algorithm (Iterative Method)
    def simplex_iterative(self):
        # check whether in standard form
        print(self.constraints)
        if all(isinstance(constraint, sp.Eq) for constraint in self.constraints):
            print("Already Standard form")
        else:
            print("Not standard form")
        

    # Simplex Algorithm (Tableau Method)
    def simplex_tableau(self):
        pass

