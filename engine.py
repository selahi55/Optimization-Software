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
        c = np.array( [float(parsed_obj_func.coeff(var)) for var in list(parsed_obj_func.free_symbols)] )
        A = [1, 2, 3]
        b = [3, 4, 6]
        # A = convert_constraints_to_matrix(self.constraints)
        # b = convert_objective_to_vector(self.)
        bounds = []

        # Determine sense
        if self.sense.lower() == 'maximization':
            c = -c  # linprog does minimization

        # Solve using scipy.optimize.linprog
        # res = scp.optimize.linprog(c, A_eq=A, b_eq=b, method='highs')
        # print(res)

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

