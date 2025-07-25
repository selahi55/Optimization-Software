# Optimization Software

A GUI application for solving optimization problems using various operations research algorithms.

## Overview

This application provides a user-friendly interface for formulating and solving optimization problems. It supports:
- Linear programming problems
- Mixed-integer programming problems
- Visualization of optimization problems in 2D and 3D
- Exporting solutions to PDF
- Saving and loading workspaces

## Installation

1. Clone this repository
2. Install the required dependencies through the requirements.txt:
   ```
   pip -r install requirements.txt
   ```

## Usage

To run the application, execute the main.py file:
```
python main.py
```

Alternatively, you can run the window.py file directly:
```
python window.py
```

## Features

- **Problem Formulation**: Input objective functions and constraints manually or load from files
- **Solution Viewing**: View the optimal solution and related information
- **Visualization**: Visualize the problem and solution in 2D or 3D
- **Settings**: Configure algorithm parameters and solver options

## Development

This project is built using:
- DearPyGUI for the user interface
- NumPy for numerical operations
- Matplotlib for visualization
- SciPy and PuLP for optimization algorithms
- Sympy for Mathematical Expression Parsing

## Troubleshooting

If the window doesn't appear:
1. Make sure all dependencies are installed correctly
2. Try running the test_dpg.py script to check if DearPyGUI is working properly
3. Check the console for any error messages
