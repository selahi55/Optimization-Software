import unittest
from utils import *

# Tests for Sympy Parser
class TestParseMathematicalExpression(unittest.TestCase):    
    def test_implicit_multiplication(self):
        test_cases = [
            ("3x", sp.parse_expr("3*x")),
            ("2y", sp.parse_expr("2*y")),
            ("3x+2y", sp.parse_expr("3*x + 2*y")),
        ]
        
        for input_expr, expected in test_cases:
            with self.subTest(input_expr=input_expr):
                result = parse_mathematical_expression(input_expr)
                self.assertEqual(result, expected)
    
    def test_explicit_multiplication(self):
        test_cases = [
            ("3*x + 2*y", sp.parse_expr("3*x + 2*y")),
            ("2*x*y", sp.parse_expr("2*x*y")),
            ("x*y*z", sp.parse_expr("x*y*z")),
        ]
        
        for input_expr, expected in test_cases:
            with self.subTest(input_expr=input_expr):
                result = parse_mathematical_expression(input_expr)
                self.assertEqual(result, expected)
    
    def test_powers_and_exponents(self):
        test_cases = [
            ("x^2", sp.parse_expr("x**2")),
            ("x**2", sp.parse_expr("x**2")),
            ("2x^2 + 3y^2", sp.parse_expr("2*x**2 + 3*y**2")),
        ]
        
        for input_expr, expected in test_cases:
            with self.subTest(input_expr=input_expr):
                result = parse_mathematical_expression(input_expr)
                self.assertEqual(result, expected)
    
    def test_mixed_operations(self):
        """Test expressions with mixed mathematical operations."""
        test_cases = [
            ("3x + 2y - 5z", sp.parse_expr("3*x + 2*y - 5*z")),
            ("x + y + z", sp.parse_expr("x + y + z")),
            ("2x*3y", sp.parse_expr("6*x*y")),
        ]
        
        for input_expr, expected in test_cases:
            with self.subTest(input_expr=input_expr):
                result = parse_mathematical_expression(input_expr)
                self.assertEqual(result, expected)
    
    def test_mathematical_functions(self):
        """Test expressions with mathematical functions."""
        test_cases = [
            ("sin(x)", sp.parse_expr("sin(x)")),
            ("cos(2x)", sp.parse_expr("cos(2*x)")),
            ("log(x) + exp(y)", sp.parse_expr("log(x) + exp(y)")),
        ]
        
        for input_expr, expected in test_cases:
            with self.subTest(input_expr=input_expr):
                result = parse_mathematical_expression(input_expr)
                self.assertEqual(result, expected)
    
    def test_complex_expressions(self):
        """Test complex mathematical expressions."""
        test_cases = [
            ("3x^2 + 2xy + y^2", sp.parse_expr("3*x**2 + 2*x*y + y**2")),
            ("(x+y)^2", sp.parse_expr("(x + y)**2")),
            ("sqrt(x^2 + y^2)", sp.parse_expr("sqrt(x**2 + y**2)")),
        ]
        
        for input_expr, expected in test_cases:
            with self.subTest(input_expr=input_expr):
                result = parse_mathematical_expression(input_expr)
                self.assertEqual(result, expected)
    
    def test_edge_cases(self):
        """Test edge cases and simple expressions."""
        test_cases = [
            ("", sp.parse_expr("0")),  # Empty string
            ("5", sp.parse_expr("5")),  # Just a number
            ("x", sp.parse_expr("x")),  # Single variable
        ]
        
        for input_expr, expected in test_cases:
            with self.subTest(input_expr=input_expr):
                result = parse_mathematical_expression(input_expr)
                self.assertEqual(result, expected)
    
    def test_invalid_expressions(self):
        """Test invalid expressions that should return None."""
        invalid_expressions = [
            "3x +",  # Incomplete expression
            "((x)",  # Mismatched parentheses
            "x y z",  # Multiple variables without operators
        ]
        
        for input_expr in invalid_expressions:
            with self.subTest(input_expr=input_expr):
                result = parse_mathematical_expression(input_expr)
                self.assertIsNone(result)
    
    def test_whitespace_handling(self):
        """Test whitespace handling in expressions."""
        test_cases = [
            ("  3x + 2y  ", sp.parse_expr("3*x + 2*y")),
            ("3x+2y", sp.parse_expr("3*x + 2*y")),
            ("3 x + 2 y", sp.parse_expr("3*x + 2*y")),
        ]
        
        for input_expr, expected in test_cases:
            with self.subTest(input_expr=input_expr):
                result = parse_mathematical_expression(input_expr)
                self.assertEqual(result, expected)

class TestGetVariables(unittest.TestCase):
    """Unit tests for get_variables function."""
    
    def test_single_variable(self):
        """Test expressions with single variables."""
        test_cases = [
            ("x", ["x"]),
            ("3x", ["x"]),
            ("sin(x)", ["x"]),
            ("x^2 + 2x + 1", ["x"]),
        ]
        
        for input_expr, expected_vars in test_cases:
            with self.subTest(input_expr=input_expr):
                parsed_expr = parse_mathematical_expression(input_expr)
                result_vars = get_variables(parsed_expr)
                result_var_names = sorted([str(var) for var in result_vars])
                expected_var_names = sorted(expected_vars)
                self.assertEqual(result_var_names, expected_var_names)
    
    def test_multiple_variables(self):
        """Test expressions with multiple variables."""
        test_cases = [
            ("x + y", ["x", "y"]),
            ("3x + 2y", ["x", "y"]),
            ("x + y + z", ["x", "y", "z"]),
            ("2x*y*z", ["x", "y", "z"]),
            ("a + b + c + d", ["a", "b", "c", "d"]),
        ]
        
        for input_expr, expected_vars in test_cases:
            with self.subTest(input_expr=input_expr):
                parsed_expr = parse_mathematical_expression(input_expr)
                result_vars = get_variables(parsed_expr)
                result_var_names = sorted([str(var) for var in result_vars])
                expected_var_names = sorted(expected_vars)
                self.assertEqual(result_var_names, expected_var_names)
    
    def test_functions_with_variables(self):
        """Test functions containing variables."""
        test_cases = [
            ("sin(x)", ["x"]),
            ("cos(x) + sin(y)", ["x", "y"]),
            ("log(x) + exp(y) + sqrt(z)", ["x", "y", "z"]),
            ("sin(x)*cos(y) + tan(z)", ["x", "y", "z"]),
        ]
        
        for input_expr, expected_vars in test_cases:
            with self.subTest(input_expr=input_expr):
                parsed_expr = parse_mathematical_expression(input_expr)
                result_vars = get_variables(parsed_expr)
                result_var_names = sorted([str(var) for var in result_vars])
                expected_var_names = sorted(expected_vars)
                self.assertEqual(result_var_names, expected_var_names)
    
    def test_complex_expressions(self):
        """Test complex expressions with variables."""
        test_cases = [
            ("x^2 + 2xy + y^2", ["x", "y"]),
            ("(x + y)*(x - y)", ["x", "y"]),
            ("3x^2 + 2xy + y^2", ["x", "y"]),
        ]
        
        for input_expr, expected_vars in test_cases:
            with self.subTest(input_expr=input_expr):
                parsed_expr = parse_mathematical_expression(input_expr)
                result_vars = get_variables(parsed_expr)
                result_var_names = sorted([str(var) for var in result_vars])
                expected_var_names = sorted(expected_vars)
                self.assertEqual(result_var_names, expected_var_names)
    
    def test_constants_only(self):
        """Test expressions with constants only (no variables)."""
        test_cases = [
            ("5", []),
            ("3 + 2", []),
            ("pi + e", []),
        ]
        
        for input_expr, expected_vars in test_cases:
            with self.subTest(input_expr=input_expr):
                parsed_expr = parse_mathematical_expression(input_expr)
                result_vars = get_variables(parsed_expr)
                result_var_names = sorted([str(var) for var in result_vars])
                expected_var_names = sorted(expected_vars)
                self.assertEqual(result_var_names, expected_var_names)
    
    def test_none_input(self):
        """Test get_variables with None input."""
        result = get_variables(None)
        self.assertEqual(result, [])
    
    def test_empty_expression(self):
        """Test get_variables with empty expression."""
        parsed_expr = parse_mathematical_expression("")
        result_vars = get_variables(parsed_expr)
        self.assertEqual(result_vars, [])
    
    def test_different_variable_names(self):
        """Test different variable naming conventions."""
        test_cases = [
            ("alpha + beta", ["alpha", "beta"]),
            ("x1 + x2", ["x1", "x2"]),
            ("X + Y", ["X", "Y"]),
        ]
        
        for input_expr, expected_vars in test_cases:
            with self.subTest(input_expr=input_expr):
                parsed_expr = parse_mathematical_expression(input_expr)
                result_vars = get_variables(parsed_expr)
                result_var_names = sorted([str(var) for var in result_vars])
                expected_var_names = sorted(expected_vars)
                self.assertEqual(result_var_names, expected_var_names)
    
    def test_mathematical_constants(self):
        """Test expressions with mathematical constants."""
        test_cases = [
            ("pi*x + e*y", ["x", "y"]),
            ("2*pi*r", ["r"]),
            ("x + I*y", ["x", "y"]),  # I is imaginary unit
        ]
        
        for input_expr, expected_vars in test_cases:
            with self.subTest(input_expr=input_expr):
                parsed_expr = parse_mathematical_expression(input_expr)
                result_vars = get_variables(parsed_expr)
                result_var_names = sorted([str(var) for var in result_vars])
                expected_var_names = sorted(expected_vars)
                self.assertEqual(result_var_names, expected_var_names)

class TestEdgeCases(unittest.TestCase):
    """Unit tests for edge cases and error conditions."""
    
    def test_long_expressions(self):
        """Test very long expressions."""
        input_expr = "x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10"
        expected_vars = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10"]
        
        parsed_expr = parse_mathematical_expression(input_expr)
        result_vars = get_variables(parsed_expr)
        result_var_names = sorted([str(var) for var in result_vars])
        expected_var_names = sorted(expected_vars)
        
        self.assertEqual(result_var_names, expected_var_names)
    
    def test_case_sensitivity(self):
        """Test case sensitivity in variable names."""
        input_expr = "X + x + Y + y"
        expected_vars = ["X", "x", "Y", "y"]
        
        parsed_expr = parse_mathematical_expression(input_expr)
        result_vars = get_variables(parsed_expr)
        result_var_names = sorted([str(var) for var in result_vars])
        expected_var_names = sorted(expected_vars)
        
        self.assertEqual(result_var_names, expected_var_names)
    
    def test_repeated_variables(self):
        """Test expressions with repeated variables."""
        input_expr = "x + x + x"
        expected_vars = ["x"]  # Should only appear once
        
        parsed_expr = parse_mathematical_expression(input_expr)
        result_vars = get_variables(parsed_expr)
        result_var_names = sorted([str(var) for var in result_vars])
        expected_var_names = sorted(expected_vars)
        
        self.assertEqual(result_var_names, expected_var_names)

