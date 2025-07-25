import dearpygui.dearpygui as dpg
from utils import *
from engine import Engine

class Window:
    def __init__(self, screen_width, screen_height):
        # context
        dpg.create_context()

        # viewport setup
        dpg.create_viewport(title="General-Purpose Optimization Software", width=screen_width, height=screen_height)
        dpg.setup_dearpygui()

        # Setup
        self.engine = Engine()
        self.constraint_count = 1

    def problem_submit(self):
        self.sense = dpg.get_value("sense")
        self.obj_func = dpg.get_value("obj_func")
        self.constraints = [dpg.get_value(f"constraint_{i+1}") for i in range(self.constraint_count)]
        print(self.sense)
        print(self.obj_func)
        print(self.constraints)

    def add_constraint(self):
        self.constraint_count +=1
        # Adds a new constraint above the Add Constraint button
        dpg.add_input_text(hint=f"Constraint {self.constraint_count} (e.g. 2x + 3y <= 4)", 
                           width=500, tag=f"constraint_{self.constraint_count}",
                           parent="constraint_area",
                           before="constraint_button")

    # Mathematical problem formulation 
    def setup_problem_formulation(self):
        dpg.add_spacer(height=10)
        dpg.add_text("Problem Formulation:", color="red")
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
            
    def solve(self):
        pass

    def run(self):
        # main window
        with dpg.window(label="Main Window", tag="main_window"):            
            with dpg.group(horizontal=True):
                dpg.add_button(label="Load Problem", callback=lambda: print("Load Problem clicked"))
                
            dpg.add_separator()
            self.setup_problem_formulation()
        
            dpg.add_separator()
            self.setup_solve_menu()

        # more setup
        dpg.set_primary_window("main_window", True)
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()