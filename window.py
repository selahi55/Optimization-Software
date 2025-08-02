import dearpygui.dearpygui as dpg
from utils import *
from engine import Engine

class Window:
    def __init__(self, screen_width, screen_height):
        # window setup
        dpg.create_context()
        dpg.create_viewport(
            title="General-Purpose Optimization Software",
            width=screen_width,
            height=screen_height,
            small_icon="assets/icon.ico",
            large_icon="assets/icon.ico"
        )
        dpg.setup_dearpygui()

        self.constraint_count = 1

    # ===PROBLEM SUBMIT AND VALIDATION===

    def problem_submit(self):
        # Clear all existing errors
        self.hide_error()

        self.sense = dpg.get_value("sense")
        self.obj_func = dpg.get_value("obj_func")
        self.constraints = [dpg.get_value(f"constraint_{i+1}") for i in range(self.constraint_count) if dpg.get_value(f"constraint_{i+1}")]

        try:
            validation_result = self.problem_validation(self.sense, self.obj_func, self.constraints)
            if not validation_result:
                return 
            self.engine = Engine(self.sense, self.obj_func, self.constraints)
            self.engine.simplex_scipy()
        except Exception as e:
            self.show_error(f"Solving failed: {str(e)}")

    def problem_validation(self, sense, obj_func, constraints):
        # Validation of objective function
        if not obj_func.strip():
            self.show_error("Objective function is required")
            return False
            
        parsed_obj = parse_mathematical_expression(obj_func)
        if parsed_obj is None:
            self.show_error("Invalid objective function")
            return False

        # Validation of constraints
        if not constraints:
            self.show_error("At least one constraint is required")
            return False
            
        for i, constraint in enumerate(constraints, 1):
            print(constraint, i)
            if constraint:  # Only validate non-empty constraints
                parsed_constraint = parse_mathematical_expression(constraint)
                
                if parsed_constraint is None:
                    self.show_error(f"Invalid constraint {i}: {constraint}")
                    return False
        
        self.hide_error()
        return True
        
    def add_constraint(self):
        self.constraint_count +=1
        # Adds a new constraint above the Add Constraint button
        dpg.add_input_text(hint=f"Constraint {self.constraint_count} (e.g. 2x + 3y <= 4)", 
                           width=500, tag=f"constraint_{self.constraint_count}",
                           parent="constraint_area",
                           before="constraint_button")

    # ===SETUP MENU FUNCTIONS===

    def setup_problem_formulation(self):
        # Error display area (initially hidden)
        with dpg.group(tag="error_group"):
            dpg.add_text("", tag="error_text", color=(255, 0, 0))  # Red text
            dpg.add_spacer(height=5)
        dpg.hide_item("error_group")

        dpg.add_spacer(height=10)
        dpg.add_text("Manual Problem Formulation:", color=(0, 128, 0))
        with dpg.group(horizontal=True):
            dpg.add_text("Objective Function:")
            with dpg.group(horizontal=True):
                dpg.add_combo(["Minimize", "Maximize"], default_value="Minimize", width=100, tag="sense")
                dpg.add_input_text(hint="Enter your objective function (e.g. 2x + 3y)", width=500, tag="obj_func")
        
        dpg.add_separator()
        with dpg.group(tag="constraint_area"):
            dpg.add_text("Constraints:")
            dpg.add_input_text(hint="Constraint 1 (e.g. 2x + 3y <= 4)", width=500, tag="constraint_1")
            dpg.add_button(label="+ Add Constraint", callback=self.add_constraint, tag="constraint_button")
            
    def setup_solve_menu(self):
        with dpg.group(horizontal=True):
            dpg.add_button(label="Solve", callback=self.problem_submit)
            dpg.add_button(label="Visualize", callback=lambda: print("Visualize clicked"))
            
    # ===ERROR HANDLING===

    def show_error(self, message):
        # Displays errors
        dpg.set_value("error_text", f"Error: {message}")
        dpg.show_item("error_group")

    def hide_error(self):
        # Hides errors
        dpg.hide_item("error_group")

    def solve(self):
        pass

    def run(self):
        # Main application window
        with dpg.window(label="Main Window", tag="main_window"):
            with dpg.tab_bar(tag="top_tabbar"):

                # Linear Programming Tab
                with dpg.tab(label="Linear Programming"):

                    dpg.add_separator()
                    self.setup_problem_formulation()
                    dpg.add_separator()
                    self.setup_solve_menu()

                # Knapsack Tab
                with dpg.tab(label="Knapsack"):
                    dpg.add_text("Knapsack solver UI coming soon...")

                # Transportation Tab
                with dpg.tab(label="Transportation"):
                    dpg.add_text("Transportation solver UI coming soon...")

        # Setup viewport
        dpg.set_primary_window("main_window", True)
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()

class Sidebar:
    def __init__(self):
        pass

class Problem:
    def __init__(self):
        pass