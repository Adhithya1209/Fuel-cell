# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 14:35:14 2023

@author: jayaraman
"""


import numpy as np
# import random
import pandas
import copy
# import matplotlib.pyplot as plt
# import tensorflow as tf

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# import time
# from dataclasses import dataclass, field
# import opem
# import keras_tuner as kt
import os
from scipy import signal
# from datetime import datetime, timedelta
# import warnings
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

import fuel_cell

#%% Equivalent circuit model - Nexa

t_start = 0
t_end = 350
step_start = 65
step_duration = 285
signal_amplitude = 33
signal_name = "I_fc"
initial_amplitude = 13

df_test = fuel_cell.create_signal(t_start, t_end, step_start, step_duration, signal_amplitude, signal_name, initial_amplitude)
nexa = fuel_cell.EquivalentCircuitRL(input_signal=df_test)
V_fc_df = nexa.output_voltage()
x = nexa.output_signal["t"]
y = nexa.output_signal["V_fc"]
nexa.plot_results(x,y,"time","V_fc","output_signal")

#%% Generic Matlab model - epac - step signal

t_start = 0
t_end = 180
step_start = 0
step_duration = 18
signal_amplitude = 1.4
signal_name = "I_fc"
input_df = fuel_cell.create_signal(t_start, t_end, step_start, step_duration, signal_amplitude, signal_name)

epac = fuel_cell.GenericMatlabModel(input_df = input_df)

test_df = epac.generate_input_signal(T = None, P_fuel = [1.35], P_air =[1.25] , Vair = None, Vfuel = None, x = None, y = None)
epac.input_df = copy.deepcopy(test_df)
V_df = epac.dynamic_response(transfer_function="on")
output_df = copy.deepcopy(epac.input_df)
epac.plot_results(output_df["t"], output_df["V_fc"], "time", "Voltage", "Voltage_output")
#epac.plot_results(output_df["I_fc"], output_df["V_fc"], "current", "Voltage", "Polarisation curve")
epac_input_df = epac.input_df
#%% Nedstack polarisation curve 
#current_dir = os.getcwd()
#input_df = pandas.read_csv("C:/IVI/Fuel_cell/utility/input_current_time.csv")
I_fc = np.linspace(0.01,200, 120)
t = np.linspace(0, 500, 120)
input_df = pandas.DataFrame()
input_df["I_fc"] = I_fc
input_df["t"] = t

nominal_parameters = {"En_nom": None,
                      "In": 133.3,
                      "Vn": 45,
                      "Tn": 338,
                      "xn": 0.99999,
                      "yn": 0.21,
                      "P_fueln": 1.5,
                      "P_airn" : 1.0,
                      "Eoc" : 65,
                      "V1" :63,
                      "N" : 65,
                      "nnom":0.55,
                      "Vairn": 297,
                      "w": 0.01,
                      "Imax": 225, 
                      "Vmin" : 37,
                      "Td" : 10,
                      "ufo2_peak_percent" : 0.65,
                      "Vu": 1.575, 
                      }

input_state = {

                "P_fuel": 1.5,
                "P_air": 1,
                "T": 338,
                "x": 0.999,
                "y": 0.21,
                "P_fuel" : 1.5,
                "P_air" : 1,
                "Vair": 297
                }

#input_df.drop(index= len(input_df)-1, inplace = True)

nedstack = fuel_cell.GenericMatlabModel(input_df = input_df, nominal_parameters = nominal_parameters)
test_df = nedstack.generate_input_signal(T = [input_state["T"]], P_fuel = [input_state["P_fuel"]], P_air = [input_state["P_air"]], 
                                         Vair = [input_state["Vair"]], Vfuel = [75], x = [input_state["x"]], 
                                         y = [input_state["y"]])
nedstack.input_df = copy.deepcopy(test_df)

nedstack.dynamic_response(x0 = -32.5, transfer_function="on")
nedstack_fc = nedstack.fuel_cell_parameters
nedstack_response = nedstack.response_df
nedstack_calc = nedstack.calculated_space

nedstack.plot_results(nedstack_response["I_fc"], nedstack_response["V_fc"], "current", "Voltage", "Polarisation curve")



#%% Nexa - Generic model- simplified model

nominal_parameters = {"En_nom": None,
                      "In": 52,
                      "Vn": 24.23,
                     
                      "Eoc" : 28.32,
                      "V1" :32.7,
                      "N" : 35,
                      "nnom":0.50,
                      "xn": 0.9995,
                      "Vairn": 2903,
                      "Imax": 100, 
                      "Vmin" : 20,
                      "Td" : 1,
                      "ufo2_peak_percent" : 0.65,
                      "Vu": 0.84805,
                      # "Tn": 326,
                      
                      # "yn": 0.21,
                      # "P_fueln": 1.5,
                      # "P_airn" : 1.0,
                      # "w": 0.01,
                      }
t_start = 0
t_end = 350
step_start = 65
step_duration = 285
signal_amplitude = 33
signal_name = "I_fc"
initial_amplitude = 13
df_test = fuel_cell.create_signal(t_start, t_end, step_start, step_duration, signal_amplitude, signal_name, initial_amplitude)
nexa_generic = fuel_cell.GenericMatlabModel(input_df = df_test, nominal_parameters=nominal_parameters)
NA = nexa_generic.calc_na()
Eoc_static = nominal_parameters["Eoc"]
I_fc = nexa_generic.input_df["I_fc"].tolist()
i0_static = nexa_generic.calc_i0()
i0 = np.full_like(I_fc, i0_static)
divide_array = np.array(np.divide(I_fc, i0))
signal_f = NA*np.log(divide_array)
num = [1.0]
den = [20/3, 1.0]
tf = signal.TransferFunction(num, den)
signal_f_t, signal_f_y, X_out = signal.lsim(tf,
                                        U =signal_f, 
                                        T =nexa_generic.input_df["t"], X0= [-0.3], interp = False)
Eoc = np.full_like(I_fc, Eoc_static)
E = Eoc - signal_f_y
Rohm = nexa_generic.nominal_parameters["Rohm"]
E = np.array(E)
I_fc = np.array(I_fc)
V_fc = E - Rohm* I_fc
V_fc = V_fc[120:]
I_fc = I_fc[120:]
t = np.array(nexa_generic.input_df["t"].tolist())
t = t[120:]
fuel_cell.plot_results(t, V_fc, "t", "V_fc", "output_signal")
#%% Nexa polarisation - generic
I_fc = np.linspace(0.01,100, 120)
t = np.linspace(0, 350, 120)
input_df = pandas.DataFrame()
input_df["I_fc"] = I_fc
input_df["t"] = t
nominal_parameters = {"En_nom": None,
                      "In": 52,
                      "Vn": 24.23,
                     
                      "Eoc" : 28.32,
                      "V1" :32.7,
                      "N" : 35,
                      "nnom":0.50,
                      "xn": 0.9995,
                      "Vairn": 2903,
                      "Imax": 100, 
                      "Vmin" : 20,
                      "Td" : 1,
                      "ufo2_peak_percent" : 0.65,
                      "Vu": 0.84805,
                      # "Tn": 326,
                      
                      # "yn": 0.21,
                      # "P_fueln": 1.5,
                      # "P_airn" : 1.0,
                      # "w": 0.01,
                      }
nexa_generic = fuel_cell.GenericMatlabModel(input_df = input_df, nominal_parameters=nominal_parameters)
NA = nexa_generic.calc_na()
Eoc_static = nominal_parameters["Eoc"]
I_fc = nexa_generic.input_df["I_fc"].tolist()
i0_static = nexa_generic.calc_i0()
i0 = np.full_like(I_fc, i0_static)
divide_array = np.array(np.divide(I_fc, i0))
signal_f = NA*np.log(divide_array)
num = [1.0]
den = [20/3, 1.0]
tf = signal.TransferFunction(num, den)
signal_f_t, signal_f_y, X_out = signal.lsim(tf,
                                        U =signal_f, 
                                        T =nexa_generic.input_df["t"], X0= [-61], interp = True)
Eoc = np.full_like(I_fc, Eoc_static)
E = Eoc - signal_f_y
Rohm = nexa_generic.nominal_parameters["Rohm"]
E = np.array(E)
I_fc = np.array(I_fc)
V_fc = E - Rohm* I_fc

t = np.array(nexa_generic.input_df["t"].tolist())

fuel_cell.plot_results(t, V_fc, "t", "V_fc", "output_signal")
#%% Test Nexa - RL
cell_parameters = {"Eoc": 30.3,
                        "R1" : 0.154,
                        "R2" : 0.1515,
                        "L": 4.94}
t_start = 0
t_end = 350
step_start = 65
step_duration = 285
signal_amplitude = 33
signal_name = "I_fc"
initial_amplitude = 13

df_test = fuel_cell.create_signal(t_start, t_end, step_start, step_duration, signal_amplitude, signal_name, initial_amplitude, plot = True)
nexa = fuel_cell.EquivalentCircuitRL(input_signal = df_test, cell_parameters = cell_parameters, test=True)
V_fc_df = nexa.output_voltage()
x = nexa.output_signal["t"]
y = nexa.output_signal["V_fc"]
#nexa.plot_results(x,y,"time","V_fc","output_signal")

#%% Steady State empirical - Nedstack PS6

nedstack = fuel_cell.SteadyStateEmprical()
nedstack.generate_input_signal(I=[5,200], T=[343], PH2=[1], PO2=[1])

response_df = nedstack.run_steady_state()