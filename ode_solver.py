# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 09:22:34 2023

@author: jayaraman
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from state_space_demo import StateSpaceModel
#%% Simple model 1 odeint
# Define the system of differential equations with variable coefficients
def model(y, t):
    print(type(y))
    y1, y2 = y
    k1 = some_function_of_y1_y2(y1, y2)  # Function to compute k1
    k2 = another_function_of_y2(y2)     # Function to compute k2
    dy1dt = -k1 * y1
    dy2dt = k1 * y1 - k2 * y2
    return [dy1dt, dy2dt]

# Define functions for variable coefficients
def some_function_of_y1_y2(y1, y2):
    # Implement your function here
    return y1 * y2

def another_function_of_y2(y2):
    # Implement your function here
    return y2 ** 2

# Initial conditions
y0 = [1.0, 0.0]


# Time points
t = np.linspace(0, 10, 100)

# Solve the system of differential equations
y = odeint(model, y0, t)

# Extract the solution for y1 and y2
y1, y2 = y[:, 0], y[:, 1]

# Plot the results
plt.figure()
plt.plot(t, y1, 'r', label='y1(t)')
plt.plot(t, y2, 'b', label='y2(t)')
plt.xlabel('Time')
plt.ylabel('y')
plt.legend()
plt.show()

#%% Simple model 2
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the ODE
def my_ode(t, y):
    dydt = -2 * t * y
    return dydt

# Set the initial condition
initial_condition = [1.0]

# Define the time span
t_span = (0, 2)

# Solve the ODE
sol = solve_ivp(my_ode, t_span, initial_condition, t_eval=np.linspace(0, 2, 100))

# Plot the solution
plt.plot(sol.t, sol.y[0], label='y(t)')
plt.xlabel('Time')
plt.ylabel('y(t)')
plt.legend()
plt.show()

#%% State space model demo 

SSM = StateSpaceModel()
cell_parameters = SSM.cell_parameters
results = SSM.ode_solver()
theta =SSM.current_theta
a_matrix = SSM.matrix_a_comp
b_matrix = SSM.matrix_b_comp
g_matrix = SSM.matrix_g_comp
current_state_variables = SSM.current_state_variables
x = SSM.x
input_df =SSM.input_df