
import numpy as np
import random
import pandas
import copy
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GaussianNoise
import time
from dataclasses import dataclass, field
import opem
import keras_tuner as kt
import os
from scipy import signal
from datetime import datetime, timedelta
import warnings
from sklearn.model_selection import train_test_split
from scipy.integrate import odeint
from scipy.integrate import solve_ivp

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
"""
1. PEM fuel cell models - ElectroChemicalDynamic (OPEM)
                            CorreaModel
                            GenericMatlabModel
                            EquivalentCircuitRL
                            
2. Data Preparation Module - SimulatedData

3. Neural network Module

"""
#%% Create input signals for the transient models

#### Base signals
def step_signal(start_time, end_time, step_time, step_amplitude, num=1000):
    """
    
    To create a base step signal. To be provided as inputs for fuel_cell model.
    This function is used by create_signal() to create a time based signal.
    
    Parameters
    ----------
    start_time : int
        signal starting time.
        
    end_time : int
        signal ending time.
        
    step_time: int
        time at which step occurs
        
    step_amplitude: float
        Amplitude of the signal after step change

    Returns
    -------
    signal: numpy array
        step signal is returned
    
    Example
    -----------
    import fuel_cell
    
    signal = fuel_cell.step_signal(0,20,10,1.4)
    """

    t = np.linspace(start_time, end_time, num)
    signal = np.where(t >= step_time, step_amplitude, 0)
    
    
    return signal

#### Generate input signal
def create_signal(t_start, t_end, step_start, step_duration, signal_amplitude, 
                  signal_name, initial_amplitude = None, number_t = 100, plot = False):
    
    """
    To create multiple step signals in the between t_start and t_stop. Uses step_signal() 
    to create a base signal in an subinterval between t_start and t_stop. The 
    step increase (amplitude) is uniform throughout the signal
    
    Parameters
    ----------
    t_time : int
        signal starting time.
        
    t_time : int
        signal ending time.
        
    step_start: int
        time at which step occurs
        
    step_duration: int
        duration for which a single step lasts
        
    step_amplitude: float
        Amplitude of the signal after step change
        
    signal_name: string
        Name of the signal
        
    initial_amplitude: float
        The initial amplitude of the signal during t_start. Default None

    Returns
    -------
    df: pandas dataframe
        signal is returned
    
    Example
    -----------
    import fuel_cell
    
    signal = fuel_cell.create_signal(0,20,0,10,1.4, "I_fc")
    """
    
    # number_steps = (t_end - step_start)/step_duration

    total_num = (t_end - t_start) * number_t

    t = np.linspace(t_start, t_end, total_num)
    df_signal = pandas.DataFrame()
    df_signal = pandas.DataFrame(columns = ["{}".format(signal_name)])
    
    signal_step = []
    step_amplitude = 0
    
    for i in range(t_start, t_end, step_duration):
        
        if not isinstance(initial_amplitude, type(None)) and i == t_start:
            
            signal_amplitude_step = initial_amplitude
            end_time = i + step_start
            num = number_t* step_start
            signal_step = step_signal(i, end_time, i, 
                                      step_amplitude=signal_amplitude_step, num = num) 
            
        else:
            signal_amplitude_step = signal_amplitude + step_amplitude
            step_amplitude = signal_amplitude_step
            end_time = i + step_duration
            num = number_t* step_duration
            signal_step = step_signal(i, end_time, i, 
                                      step_amplitude=signal_amplitude_step, num = num) 
        df_signal_step = pandas.DataFrame(signal_step, columns=["{}".format(signal_name)])
        df_signal = pandas.concat([df_signal, df_signal_step], ignore_index=True)
     
    df_signal["t"] = t
    
    if plot:
        plt.plot(df_signal["t"], df_signal["{}".format(signal_name)])
        plt.xlabel('Time')
        plt.ylabel('{}'.format(signal_name))
        plt.title('Input Signal')
        plt.grid(True)
        plt.show()
    
    return df_signal

#%% Utility functions

def plot_results(x, y, xlabel, ylabel, title):
    """
    Plot results. Does not allow subplots currently

    Parameters
    ----------
    x : list
        x axis values as list
    y : list
        y axis values as lists
    xlabel : string
        Label of x axis
    ylabel : string
        label of y axis
    title : string
        Title of the plot

    Returns
    -------
    None.

    """
    plt.plot(x, y, linestyle = '--')
    plt.xlabel('{}'.format(xlabel))
    plt.ylabel('{}'.format(ylabel))
    plt.title('{}'.format(title))
    plt.grid(True)
    # plt.yticks(ticks=[0,20,40,60,80,100])
    # plt.ylim(ymin=0)
    # plt.xlim(left=0)
    plt.show()
    
    
def create_timestamp(timestamp_list = None, number_time_steps = None, duration = 1):
    """
    Create timestamp given start and end timestamps and number of time steps

    Parameters
    ----------
    timestamp_list : list, optional
        start and end timestamps for the duration. The default is None.
    number_time_steps : int, optional
        . The default is None.
    duration : TYPE, optional
        DESCRIPTION. The default is 1.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    timestamp : TYPE
        DESCRIPTION.
    step_duration : TYPE
        DESCRIPTION.

    """
    if isinstance(number_time_steps, type(None)):
        raise ValueError("number_time_steps parameter required to generate timestamp")
        
    if isinstance(timestamp_list, type(None)):
        
        warnings.warn("Timestamp was not provided. Taking default"\
                      " timestamp values. Check results properly")
            
        start_time_str = '2023-06-01 12:30:45'
        
        start_time = datetime.strptime(
            start_time_str, '%Y-%m-%d %H:%M:%S')
        
        step_duration = timedelta(seconds = duration)
        
    else:
        for index, value in enumerate(timestamp_list):
            if index == 0:
                start_time_str = value
                
            elif index == 1:
                end_time_str = value
                
            else:
                raise ValueError("Only start and end values for"\
                                 " timestamp should be provided "\
                                "as list. Check input values "\
                                         "for timestamp")
          
        start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')
        end_time = datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S')
        step_duration = (end_time - start_time) / number_time_steps
        
    timestamp = [start_time + i * step_duration for i in range(number_time_steps)]

    return timestamp, step_duration

# Dummy input signal - Generic model
def dum_input():
    """
    

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """
    df = pandas.DataFrame()
    I_fc = np.linspace(1, 14.155, 100)
    t = np.linspace(1, 14.155, 100)
    P_fuel = np.full_like(I_fc, 0.3256)
    P_air = np.full_like(I_fc, 0.3454)
    T = np.full_like(I_fc, 315.15)
    Vfuel = np.full_like(I_fc, 14.91)
    Vair = np.full_like(I_fc, 14.91)
    
    x = np.full_like(I_fc, 0.9995)
    y = np.full_like(I_fc, 0.21)
    data = {"I_fc": I_fc,
            "P_fuel" : P_fuel,
            "P_air": P_air,
            "T" : T,
            "Vfuel": Vfuel,
            "Vair": Vair,
            "x": x,
            "y": y,
            "t": t}
    
    df = pandas.DataFrame(data)
           
    return df

def dum_input_current():
    df = pandas.DataFrame(columns = ["I", "t"])
    time = []
    t1 = np.linspace(99.5, 100.2, 100)
    t2= 100*[100.2]
    t3 = np.linspace(100.2, 100.4, 100)
    t4= 100*[100.4]
    t5 = np.linspace(100.4, 100.6, 100)
    t6 = 100*[100.6]
    t7 = np.linspace(100.6, 100.7, 100)
    t8 = 100*[100.7]
    t9 = np.linspace(100.7, 102, 100)

    time =np.append(time, t1)

    time =np.append(time, t2)
    time =np.append(time, t3)
    time =np.append(time, t4)
    time =np.append(time, t5)
    time =np.append(time, t6)
    time =np.append(time, t7)
    time =np.append(time, t8)
    time =np.append(time, t9)

    current = []
    c1= 100*[12.5]
    c2 = np.linspace(12.5, 20.5, 100)
    c3 = 100*[20.5]
    c4 = np.linspace(20.5, 24.5, 100)
    c5 = 100*[24.5]
    c6 = np.linspace(24.5, 20.5, 100)
    c7 = 100*[20.5]
    c8 = np.linspace(20.5, 13, 100)
    c9 = 100*[13]
    current = np.append(current, c1)
    current =np.append(current, c2)
    current =np.append(current, c3)
    current =np.append(current, c4)
    current =np.append(current, c5)
    current =np.append(current, c6)
    current =np.append(current, c7)
    current =np.append(current, c8)
    current =np.append(current, c9)
    
    df["I"] = current
    df["t"] = time
    
    return df
    

#%% Utility class

# update required for numpy array initialisation
@dataclass       
class DataClass:
    
    data : pandas.DataFrame = field(default_factory=pandas.DataFrame)
    scaler: StandardScaler = field(default_factory=StandardScaler)
    scaled_data : pandas.DataFrame = field(default_factory=pandas.DataFrame)
    X_train : np.ndarray = None
    y_train : np.ndarray = None
    X_test : np.ndarray = None
    y_test: np.ndarray = None
    predicted_data : np.ndarray = None

# Data class initialisation
tensor_data = DataClass()
#%% Fuel cell models

class ElectroChemicalDynamic:
    
    def __init__(self, electro_model = None, test_mode = False, 
                 input_vector = None):
        
        if isinstance(electro_model, type(None)):
            print("Type of electrochemical model is not provided. "\
                  "Running PadullesDynamic1 by default")
            
        else:
            self.electro_model = electro_model
        self.test_mode = test_mode
        self.input_vector = input_vector
        self.simulated = DataClass()
 
    def select_model(self):
        """
        

        Returns
        -------
        None.

        """
        electro_model = self.electro_model
        test_mode = self.test_mode
        input_vector = self.input_vector
        
        if electro_model == "chakraborty":
            self.simulated.data = self.run_chakraborty(test_mode, input_vector)
                
        elif electro_model == "padulles_dynamic1":
            self.simulated.data = self.run_padulles1(test_mode, input_vector)
            
        elif electro_model == "padulles_dynamic2":
            self.simulated.data = self.run_padulles2(test_mode, input_vector)
            
        elif electro_model == "padulles_hauer":
            self.simulated.data = self.run_padulles_hauer(test_mode, input_vector)
            
        elif electro_model == "padulles_amphlett":
            self.simulated.data = self.run_padulles_amphlett(test_mode, input_vector)
            
        else:
            print("Invalid model. Check the model name")
            
    def run_chakraborty(self,test_mode, input_vector):
        
        Test_Vector = {
                        "T": 1273,
                        "E0": 0.6,
                        "u":0.8,
                        "N0": 1,
                        "R": 3.28125 * 10**(-3),
                        "KH2O": 0.000281,
                        "KH2": 0.000843,
                        "KO2": 0.00252,
                        "rho": 1.145,
                        "i-start": 0.1,
                        "i-stop": 300,
                        "i-step": 0.1,
                        "Name": "Chakraborty_Test"}
        
        if test_mode or isinstance(input_vector, type(None)):
            
            input_vector = Test_Vector
            
            data=opem.Dynamic.Chakraborty.Dynamic_Analysis(
                InputMethod=input_vector,TestMode=True,PrintMode=True,
                ReportMode=True)
            
            self.input_vector = input_vector
            return data
        
        else:
            if not set(Test_Vector.keys())== set(input_vector.keys()):
                print("Invalid input vector for chakraborty model. Check the "\
                      "input_vector again")
                
            else:
                self.input_vector = input_vector
                data=opem.Dynamic.Chakraborty.Dynamic_Analysis(
                    InputMethod=input_vector,TestMode=True,PrintMode=True,
                    ReportMode=True)
                
                
        return data
    
    def run_padulles1(self,test_mode, input_vector):
        
        Test_Vector = {
                        "T": 343,
                        "E0": 0.6,
                        "N0": 88,
                        "KO2": 0.0000211,
                        "KH2": 0.0000422,
                        "tH2": 3.37,
                        "tO2": 6.74,
                        "B": 0.04777,
                        "C": 0.0136,
                        "Rint": 0.00303,
                        "rho": 1.168,
                        "qH2": 0.0004,
                        "i-start": 0,
                        "i-stop": 100,
                        "i-step": 0.1,
                        "Name": "PadullesI_Test"}
        
        
        if test_mode or isinstance(input_vector, type(None)):
            
            input_vector = Test_Vector
            
            data= opem.Dynamic.Padulles1.Dynamic_Analysis(
                InputMethod=input_vector,TestMode=True,PrintMode=True,
                ReportMode=True)
            
            self.input_vector = input_vector
            
            return data
        
        else:
            if not set(Test_Vector.keys())== set(input_vector.keys()):
                print("Invalid input vector for padulles model. Check the"\
                      " input_vector again")
                
            else:
                self.input_vector = input_vector
                data=opem.Dynamic.Padulles1.Dynamic_Analysis(
                    InputMethod=input_vector,TestMode=True,PrintMode=True,
                    ReportMode=True)
                    
        return data    
    
    def run_padulles2(self,test_mode, input_vector):
        
        Test_Vector = {
                        "T": 343,
                        "E0": 0.6,
                        "N0": 5,
                        "KO2": 0.0000211,
                        "KH2": 0.0000422,
                        "KH2O": 0.000007716,
                        "tH2": 3.37,
                        "tO2": 6.74,
                        "tH2O": 18.418,
                        "B": 0.04777,
                        "C": 0.0136,
                        "Rint": 0.00303,
                        "rho": 1.168,
                        "qH2": 0.0004,
                        "i-start": 0.1,
                        "i-stop": 100,
                        "i-step": 0.1,
                        "Name": "Padulles2_Test"}
        
        
        if test_mode or isinstance(input_vector, type(None)):
            
            input_vector = Test_Vector
            
            data= opem.Dynamic.Padulles2.Dynamic_Analysis(
                InputMethod=input_vector,TestMode=True,PrintMode=True,
                ReportMode=True)
            
            self.input_vector = input_vector
            return data
        
        else:
            if not set(Test_Vector.keys())== set(input_vector.keys()):
                print("Invalid input vector for padulles model. Check the "\
                      "input_vector again")
                
            else:
                
                data=opem.Dynamic.Padulles2.Dynamic_Analysis(
                    InputMethod=input_vector,TestMode=True,PrintMode=True,
                    ReportMode=True)
                
                self.input_vector = input_vector
                
        return data    
    
    def run_padulles_hauer(self, test_mode, input_vector):
        
        Test_Vector = {
                        "T": 343,
                        "E0": 0.6,
                        "N0": 5,
                        "KO2": 0.0000211,
                        "KH2": 0.0000422,
                        "KH2O": 0.000007716,
                        "tH2": 3.37,
                        "tO2": 6.74,
                        "t1": 2,
                        "t2": 2,
                        "tH2O": 18.418,
                        "B": 0.04777,
                        "C": 0.0136,
                        "Rint": 0.00303,
                        "rho": 1.168,
                        "qMethanol": 0.0002,
                        "CV": 2,
                        "i-start": 0.1,
                        "i-stop": 100,
                        "i-step": 0.1,
                        "Name": "Padulles_Hauer_Test"}
        
        if test_mode or isinstance(input_vector, type(None)):
            
            input_vector = Test_Vector
            
            data= opem.Dynamic.Padulles_Hauer.Dynamic_Analysis(
                InputMethod=input_vector,TestMode=True,PrintMode=True,
                ReportMode=True)
            
            self.input_vector = input_vector
            return data
        
        else:
            if not set(Test_Vector.keys())== set(input_vector.keys()):
                print("Invalid input vector for padulles model. Check the "\
                      "input_vector again")
                
            else:
                
                data=opem.Dynamic.Padulles_Hauer.Dynamic_Analysis(
                    InputMethod=input_vector,TestMode=True,PrintMode=True,
                    ReportMode=True)
                
                self.input_vector = input_vector
                
        return data    
    
    def run_padulles_amphlett(self,test_mode, input_vector):
        
        Test_Vector = {
                        "A": 50.6,
                        "l": 0.0178,
                        "lambda": 23,
                        "JMax": 1.5,
                        "T": 343,
                        "N0": 5,
                        "KO2": 0.0000211,
                        "KH2": 0.0000422,
                        "KH2O": 0.000007716,
                        "tH2": 3.37,
                        "tO2": 6.74,
                        "t1": 2,
                        "t2": 2,
                        "tH2O": 18.418,
                        "rho": 1.168,
                        "qMethanol": 0.0002,
                        "CV": 2,
                        "i-start": 0.1,
                        "i-stop": 75,
                        "i-step": 0.1,
                        "Name": "Padulles_Amphlett_Test"}
        
        if test_mode or isinstance(input_vector, type(None)):
            
            input_vector = Test_Vector
            
            data= opem.Dynamic.Padulles_Amphlett.Dynamic_Analysis(
                InputMethod=input_vector,TestMode=True,PrintMode=True,
                ReportMode=True)
            
            self.input_vector = input_vector
            return data
        
        else:
            if not set(Test_Vector.keys())== set(input_vector.keys()):
                print("Invalid input vector for padulles model. Check the "\
                      "input_vector again")
                
            else:
                
                data=opem.Dynamic.Padulles_Amphlett.Dynamic_Analysis(
                    InputMethod=input_vector,TestMode=True,PrintMode=True,
                    ReportMode=True)
                
                self.input_vector = input_vector
                
                return data 
            
            
class CorreaModel:
    # Redefine init with **kwargs
    def __init__(self, current_lower,current_upper, number_of_cells = 32,
                 cell_area = 64, temperature = 333, 
                 pressure_o2 = 0.2095, pressure_h2=1, length = 0.0178):
        
        self.current_lower = current_lower
        self.current_upper = current_upper
        
        self.input_parameters = {"current_fuel_cell": current_lower,
                                 "number_of_cells": number_of_cells,
                                 "cell_area": cell_area,
                                 "MEA_length": length,
                                 "temperature": temperature,
                                 "pressure_o2": pressure_o2,
                                 "pressure_h2": pressure_h2,
                                 }
        
        self.model_parameters = {"equivalent_contact_resistance": 0.0003,
                            "empirical_B": 0.016,
                            "empirical_e1": -0.948,
                            "empirical_e2": 0.00312,
                            "empirical_e3": 0.000074,
                            "empirical_e4": -0.000187,
                            "empirical_sci": 23,
                            "no-load-current_density": 3,
                            "maximum_current_density": 469,
                            "equivalent_circuit_capacitance": 3}

    
    def E_nernst(self):
        
        T = self.input_parameters["temperature"]
        P_h2 = self.input_parameters["pressure_h2"]
        P_o2 = self.input_parameters["pressure_o2"]
        
        E_nernst = 1.229 - 0.00085*(T- 298.15) + 0.0000431*T*(np.log(P_h2) + 0.5*np.log(P_o2))
        
        return E_nernst
    
    def activation_voltage_drop(self):
        
        model_parameters = copy.deepcopy(self.model_parameters)
        T = self.input_parameters["temperature"]
        I_fc = self.input_parameters["current_fuel_cell"]
        E1 = model_parameters["empirical_e1"]
        E2 = model_parameters["empirical_e2"]
        E3 = model_parameters["empirical_e3"]
        E4 = model_parameters["empirical_e4"]
        P_o2 = self.input_parameters["pressure_o2"]
        c_o2 = P_o2/(5.08*np.power(10,6)*np.exp(-498/T))
        return c_o2
        V_act = -(E1 + E2*T + E3*T*np.log(c_o2) + E4*T*np.log(I_fc))
        
        return V_act
    
    def  ohmic_voltage_drop(self):
        
        model_parameters = copy.deepcopy(self.model_parameters)
        sci = model_parameters["empirical_sci"]
        i_fc = self.input_parameters["current_fuel_cell"]
        A = self.input_parameters["cell_area"]
        T = self.input_parameters["temperature"]
        l = self.input_parameters["MEA_length"]
        R_c = model_parameters["equivalent_contact_resistance"]
        
        rho_m = (181.6 * (1 + 0.03 * (i_fc/A) + 0.062*np.power(T/303, 2)* np.power(i_fc/A,2.5)))/(sci - 0.634 - 3*i_fc/A)*np.exp(4.18*(T-303)/T)
        
        R_m = rho_m *l/A
        V_ohm = i_fc* (R_m + R_c)
        
        return V_ohm
    
    def concentration_voltage_drop(self):
        
        model_parameters = copy.deepcopy(self.model_parameters)
        B = model_parameters["empirical_B"]
        J = model_parameters["no-load-current_density"]
        J_max = model_parameters["maximum_current_density"]
        
        V_con = -B * np.log(1- J/J_max)
        
        return V_con
    
    def fuel_cell_voltage(self):
        
        E_nernst = self.E_nernst()
        V_act = self.activation_voltage_drop()
        V_ohm = self.ohmic_voltage_drop()
        V_con = self.concentration_voltage_drop()
        
        V_fc = E_nernst - V_act - V_ohm - V_con
        
        return V_fc
        
    def set_range(self, model_parameter, percentage):
        
        model_lower = model_parameter *(100-percentage)/100
        model_upper = model_parameter* (100+percentage)/100
        
        return model_lower, model_upper
    
    def set_parameter_range(self):
        
        model_parameters = copy.deepcopy(self.model_parameters)
        ranged_model_parameters = dict()
        
        for key in model_parameters.keys():
            range_10 = ["maximum_current_density", "empirical_e1", "empirical_e2",
                        "empirical_e3", "empirical_e4"]
            range_15 = ["equivalent_contact_resistance", "empirical_B"]
            
            if key in range_10:
                percentage = 10
                
            elif key == "no-load-current_density":
                percentage = 25
                
            elif key in range_15:
                percentage = 15
                
            elif key == "empirical_sci":
                upper_value = 24
                lower_value = 15
                ranged_model_parameters["upper_{}".format(key)] = upper_value
                ranged_model_parameters["lower_{}".format(key)] = lower_value
                continue
            
            model_parameter =model_parameters[key]
            lower_value, upper_value = self.set_range(model_parameter, percentage)
            ranged_model_parameters["upper_{}".format(key)] = upper_value
            ranged_model_parameters["lower_{}".format(key)] = lower_value
            
        return ranged_model_parameters
                
    def generate_random_distribution(self):
        
        model_parameters = copy.deepcopy(self.model_parameters)
        ranged_model_parameters = self.set_parameter_range()
        
        for key in model_parameters.keys():
            lower_value = ranged_model_parameters["lower_{}".format(key)]
            upper_value = ranged_model_parameters["upper_{}".format(key)]
            x = []
            for i in range(1000):
                x.append(random.uniform(lower_value, upper_value))

            model_parameters[key] = x
        i_fc_dist = []
        
        for i in range(1000):
            i_fc_dist.append(random.uniform(self.current_lower, self.current_upper))
            
        i_df = pandas.DataFrame(i_fc_dist, columns=["i_fc"])

        ranged_df = pandas.DataFrame.from_dict(model_parameters, 
                                               orient='index').transpose()
        ranged_df_i = pandas.concat([ranged_df, i_df], axis =1)
        
        return ranged_df_i
    
    def compute_voltage(self):
        
        ranged_df_i = self.generate_random_distribution()
        V_fc = []
        for index, row in ranged_df_i.iterrows():
            model_parameters = ranged_df_i.loc[index].to_dict()
            self.model_parameters = copy.deepcopy(model_parameters)
            self.input_parameters["current_fuel_cell"] = model_parameters["i_fc"]
            V_fc.append(self.fuel_cell_voltage())
            
        V_fc_df = pandas.DataFrame(V_fc, columns = ["v_fc"])
        ranged_df_output = pandas.concat([ranged_df_i, V_fc_df], axis =1)
        
        return ranged_df_output
    
    def compute_voltage_2(self):
        
        i_fc = np.linspace(self.current_lower, self.current_upper, 1000)
        V_fc = []
        for item in i_fc:
            self.input_parameters["current_fuel_cell"] = item
            V_fc.append(self.fuel_cell_voltage())
            
        V_fc_df = pandas.DataFrame(V_fc, columns = ["v_fc"])
        i_df = pandas.DataFrame(i_fc, columns = ["i_fc"])
        
        ranged_df_output = pandas.concat([i_df, V_fc_df], axis =1)
        
        return(ranged_df_output)
 
    def visualize_df(self):
        
        ranged_df_output = self.compute_voltage_2()
    
        plt.plot(ranged_df_output["i_fc"], ranged_df_output["v_fc"])
        plt.title("Polarization Curve - Static")
        plt.xlabel("current_fc")
        plt.ylabel("Voltage_fc")
        plt.show()
        
    def neural_network(self):
        
        ranged_df_output = self.compute_voltage()
        input_df = copy.deepcopy(ranged_df_output)
        X = input_df.drop('v_fc', axis =1)
        y = ranged_df_output['v_fc']
        
        model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(11,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
        ])
        start_time = time.time()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                            random_state=42)
        
        model.compile(loss='mean_squared_error', optimizer='adam',
                      metrics=['mean_absolute_error'])
        
        model.fit(X_train, y_train, epochs=50, batch_size=32, 
                  validation_data=(X_test, y_test))
        
        i_fc = np.linspace(self.current_lower, self.current_upper, 1000)
        V_fc_predicted = []
        for item in i_fc:
            data = copy.deepcopy(self.model_parameters)
            data["i_fc"] = item
            df = pandas.DataFrame(data, index =[0])
            V_fc_predicted.append(model.predict(df))
        
        end_time = time.time()
        training_time = end_time - start_time
        print(training_time)
        V_fc_predicted = np.array(V_fc_predicted)
        V_fc_predicted = np.reshape(V_fc_predicted, (1000,))
        V_fc_df = pandas.DataFrame(V_fc_predicted, columns = ["v_fc"])
        plt.plot(ranged_df_output["i_fc"], V_fc_df["v_fc"])
        
        plt.title("Polarization Curve - Static")
        
        model.summary()
        plt.xlabel("current_fc")
        plt.ylabel("Voltage_fc")
        plt.show()
 
            
class GenericMatlabModel:     
    
    def __init__(self, nominal_parameters = None, input_df = None, model_type = "detail"):
        if isinstance(nominal_parameters, type(None)):
            
            # EPAC 500
            self.nominal_parameters = {"En_nom": None,
                                  "In": 8.128,
                                  "Vn": 50.28,
                                  "Tn": 315.45,
                                  "xn": 0.9995,
                                  "yn": 0.21,
                                  "P_fueln": 1.35,
                                  "P_airn" : 1.25,
                                  "Eoc" : 65.7,
                                  "V1" : 58.4,
                                  "N" : 65,
                                  "nnom":0.5883,
                                  "Vairn": 14.91,
                                  "w": 1,
                                  "Imax": 14.155, 
                                  "Vmin" : 45.707,
                                  "Td" : 1,
                                  "ufo2_peak" : 67,
                                  "ufo2_peak_percent" : 0.65,
                                  "Vu": 2.5, 
                                  }
        else:
            self.nominal_parameters = copy.deepcopy(nominal_parameters)
        
        self.constants = {
                      "faradays": 96485,
                      "moving_electrons": 2,

                      "gas_constant" : 8.3145,
                      "boltzmann": 1.38 * (10** (-23)),
                      "plancks": 6.626* (10**(-34)),

                      "deltah": 241.83 *( 10 **(3))}
        
        self.current_state = {
                        "I_fc": None,
                        "P_fuel": None,
                        "P_air": None,
                        "T": None,
                        "Vfuel" : None,
                        "Vair": None,
                        "x": None,
                        "y": None,
                        "PH2" : None,
                        "PO2" : None,
                        "UfH2" : None,
                        "UfO2" : None,  
                        }
        
        if not isinstance(input_df , type(None)):
            self.input_df = copy.deepcopy(input_df)
              
        else:
            self.input_df = self.dum_input()
    
        self.fuel_cell_parameters = None
        self.calculated_space = pandas.DataFrame() 
        self.response_df = pandas.DataFrame() 
        self.En_vector = []
        self.PH2_vector = []
        self.PO2_vector = []
        self.UfH2_vector = []
        self.UfO2_vector = []            
        self.PH2O_vector = []
        
        
     #### Nominal/ Calculated Values 
        # Nominal hydrogen flow rate
    def hydro_utilisation(self, nom = True):
        """
        

        Parameters
        ----------
        nom : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        Ufh2 : TYPE
            DESCRIPTION.

        """
        F = self.constants["faradays"]
        
        if nom:
            nnom = self.nominal_parameters["nnom"]
            deltah = self.constants["deltah"]
            N = self.nominal_parameters["N"]
            Vn = self.nominal_parameters["Vn"]
            z = self.constants["moving_electrons"]
            
            Ufh2 = (nnom * deltah *N)/(z*F*Vn)
                  
        else:
            R = self.constants["gas_constant"]
            T = self.current_state["T"]
            N = self.nominal_parameters["N"]
            I_fc = self.current_state["I_fc"]
            z = self.constants["moving_electrons"]
            P_fuel = self.current_state["P_fuel"] * 1.01325 * (10 ** (5))
            Vfuel = self.current_state["Vfuel"]
            x = self.current_state["x"]
            
            Ufh2 = (60000*R * T * I_fc*N)/(z * P_fuel *Vfuel * x *F)
        self.current_state["UfH2"] = Ufh2
        
        
    # Nominal oxygen flow rate
    def oxygen_utilisation(self, nom = True):
        """
        

        Parameters
        ----------
        nom : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        Ufo2 : TYPE
            DESCRIPTION.

        """
        R = self.constants["gas_constant"]
        z = self.constants["moving_electrons"]
        N = self.nominal_parameters["N"]
        F = self.constants["faradays"]
        
        if nom:
        
            T = self.nominal_parameters["Tn"]
            I = self.nominal_parameters["In"]
            P_air = self.nominal_parameters["P_airn"]* 1.01325 * (10 ** (5))
            Vair = self.nominal_parameters["Vairn"]
            y = self.nominal_parameters["yn"]
            
            Ufo2 = (60000* R * T * I*N)/(2 * z * P_air *Vair * y*F)
            self.nominal_parameters["ufo2n"] = Ufo2
            
        else:
            
            T = self.current_state["T"]
            I_fc = self.current_state["I_fc"]
            P_air = self.current_state["P_air"] * 1.01325 * (10 ** (5))
            Vair = self.current_state["Vair"]
            y = self.current_state["y"]
            
            Ufo2 = (60000* R * T * I_fc*N)/(2 * z * P_air *Vair * y* F)
        self.current_state["UfO2"] = Ufo2
            
            
        
    
    def calc_pressure(self, nom = True, util=True):
        """
        

        Parameters
        ----------
        nom : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        PH2 : TYPE
            DESCRIPTION.
        PO2 : TYPE
            DESCRIPTION.

        """
        if nom:
            
            x = self.nominal_parameters["xn"]
            y = self.nominal_parameters["yn"]
            if util:
                self.hydro_utilisation()
                self.oxygen_utilisation()
                self.adaptive_control_input()
            
            P_air = self.nominal_parameters["P_airn"]* 1.01325 * (10 ** (5))
            P_fuel = self.nominal_parameters["P_fueln"]* 1.01325 * (10 ** (5))
            
        else:
            x = self.current_state["x"]
            y = self.current_state["y"]
            if util:
                self.hydro_utilisation(nom = False)
                self.oxygen_utilisation(nom = False)
                self.adaptive_control_input()
            
            P_air = self.current_state["P_air"]* 1.01325 * (10 ** (5))
            P_fuel = self.current_state["P_fuel"]* 1.01325 * (10 ** (5))
        
        Ufh2 = self.current_state["UfH2"]
        Ufo2 = self.current_state["UfO2"]
        self.UfO2_vector.append(Ufo2)
        self.UfH2_vector.append(Ufh2)
        
        PH2 = x * (1 - Ufh2)*P_fuel
        self.PH2_vector.append(PH2)
        
        PO2 = y * (1 - Ufo2)*P_air
        self.PO2_vector.append(PO2)
        self.current_state["PH2"] = PH2   
        self.current_state["PO2"] = PO2   
        return PH2, PO2
    
    def calc_pressure_h2o(self, nom = True):
        """
        

        Parameters
        ----------
        nom : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        PH2O : TYPE
            DESCRIPTION.

        """
        w = self.nominal_parameters["w"]
        if nom:
            
            y = self.nominal_parameters["yn"]
            Ufo2 = self.current_state["UfO2"]
            P_air = self.nominal_parameters["P_airn"]* 1.01325 * (10 ** (5))
            
        else:
            y = self.current_state["y"]
            Ufo2 = self.current_state["UfO2"]
            P_air = self.current_state["P_air"]* 1.01325 * (10 ** (5))
            
        PH2O = (w + 2* y * Ufo2) * P_air
        self.PH2O_vector.append(PH2O)
        
        return PH2O
    
    #### Fuel cell parameter calculation
    def calc_k(self):
        """
        

        Returns
        -------
        K : TYPE
            DESCRIPTION.

        """
        Vu = self.nominal_parameters["Vu"]
        ufo2n = self.current_state["UfO2"]
        
        try:
            ufo2_peak_percent = self.nominal_parameters["ufo2_peak_percent"]
            ufo2_peak = ufo2_peak_percent * ufo2n
            
        except KeyError:
            ufo2_peak = self.nominal_parameters["ufo2_peak"]
        
        Kc = self.calc_kc()
        
        K = Vu/(Kc* (ufo2_peak - ufo2n))
        
        return K
             
    def calc_kc(self):
        """
        

        Returns
        -------
        Kc : TYPE
            DESCRIPTION.

        """
        Eoc = self.nominal_parameters["Eoc"]
        En_nom = self.calc_en()
        
        Kc = Eoc/ En_nom
        
        return Kc
    
    def calc_en(self, nom = True):
        """
        

        Parameters
        ----------
        nom : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        En : TYPE
            DESCRIPTION.

        """
        z = self.constants["moving_electrons"]
        F = self.constants["faradays"]
        R = self.constants["gas_constant"]
        
        if nom:
            T = self.nominal_parameters["Tn"] 
            PH2, PO2 = self.calc_pressure()
            PH2O = self.calc_pressure_h2o()
            
        else:
            T = self.nominal_parameters["Tn"]
            
            PH2, PO2 = self.calc_pressure(nom = False)
            
            PH2O = self.calc_pressure_h2o(nom = False)
        
        
        # PH2 = PH2/(1.01325 * (10 ** (5)))
        # PO2 = PO2/(1.01325 * (10 ** (5)))
        # PH2O = PH2O/(1.01325 * (10 ** (5)))
        
        if T>373:
            
            En = 1.229 + (T - 298)*(-44.43/(z*F)) + ((R * T)/(z* F))* np.log((PH2 * (PO2 ** (1/2)))/ PH2O)
            
        else:
            
            En= 1.229 + (T - 298)*(-44.43/(z*F)) + ((R * T)/(z* F))* np.log(PH2 * (PO2 ** (1/2)))
        
        self.En_vector.append(En)
        
        return En
    
    def calc_i0(self):
        """
        

        Returns
        -------
        i0 : TYPE
            DESCRIPTION.

        """
        V1 = self.nominal_parameters["V1"]
        Vn = self.nominal_parameters["Vn"]
        In = self.nominal_parameters["In"]
        Eoc = self.nominal_parameters["Eoc"]
        NA = self.calc_na()
        
        Rohm = (V1 - Vn - (NA * np.log(In)))/ (In - 1)
        
        self.nominal_parameters["Rohm"] = Rohm
        i0 = np.exp((V1 - Eoc + Rohm)/NA)
        
        return i0
    
    def calc_na(self):
        """
        

        Returns
        -------
        NA : TYPE
            DESCRIPTION.

        """
        V1 = self.nominal_parameters["V1"]
        Vn = self.nominal_parameters["Vn"]
        Imax = self.nominal_parameters["Imax"]
        Vmin = self.nominal_parameters["Vmin"]
        In = self.nominal_parameters["In"]
        
        NA = ((V1- Vn)*(Imax - 1) - (V1 - Vmin)*(In - 1))/ ((np.log(In))* (Imax -1) - (np.log(Imax)* (In - 1)))
        
        return NA
    
    def calc_activation_energy(self):
        """
        

        Returns
        -------
        delta_G : TYPE
            DESCRIPTION.

        """
        F = self.constants["faradays"]
        k = self.constants["boltzmann"]
        PH2n, PO2n = self.calc_pressure()
       
        h = self.constants["plancks"]
        R = self.constants["gas_constant"]
        Tn = self.nominal_parameters["Tn"]
        i0 = self.calc_i0()
        
        K1 = (2 * F* k*(PH2n +PO2n))/(h*R)
        delta_G = (-R) * Tn * np.log((i0/K1)) 
        
        return delta_G
    
    def calc_alpha(self):
        """
        

        Returns
        -------
        a : TYPE
            DESCRIPTION.

        """
        N = self.nominal_parameters["N"]
        R = self.constants["gas_constant"]
        Tn = self.nominal_parameters["Tn"]
        z = self.constants["moving_electrons"]
        F = self.constants["faradays"]
        NA = self.calc_na()
        
        a = (N * R * Tn)/ (z * F * NA)
        
        return a
    
    def calc_fc_parameters(self):
        """
        

        Returns
        -------
        fuel_cell_parameters : TYPE
            DESCRIPTION.

        """
        NA = self.calc_na()
        i0 = self.calc_i0()
        a = self.calc_alpha()
        delta_G = self.calc_activation_energy()
        Kc = self.calc_kc()
        K = self.calc_k()
        
        fuel_cell_parameters = {"alpha": a,
                                "activation_energy" : delta_G,
                                "Kc": Kc,
                                "K": K,
                                "NA" : NA,
                                "i0" : i0}
        
        self.fuel_cell_parameters = fuel_cell_parameters
        
        self.reset_calc_vectors()
      
        return fuel_cell_parameters
     
    #### Dynamic parameter calculation   
    def dynamic_i0(self):
        """
        

        Returns
        -------
        i0 : TYPE
            DESCRIPTION.

        """
        z = self.constants["moving_electrons"]
        F = self.constants["faradays"]
        k = self.constants["boltzmann"]
        R = self.constants["gas_constant"]
        h = self.constants["plancks"]
        delta_G = self.fuel_cell_parameters["activation_energy"]
        T = self.current_state["T"]
        PH2 = self.current_state["PH2"]
        PO2 = self.current_state["PO2"]
        i0 = (z* F * k *(PH2 + PO2))/(R*h) * np.exp(-delta_G/ (R*T))
        
        return i0
    
    def oxygen_depletion(self, En):
        
        ufo2 = self.current_state["UfO2"]
        ufo2n = self.nominal_parameters["ufo2n"]
        K = self.fuel_cell_parameters["K"]
        En = En - K * (ufo2 - ufo2n)
        
        return En
    
    def dynamic_block_calculation(self):
        """
        

        Returns
        -------
        None.

        """
        fuel_cell_parameters = self.calc_fc_parameters()
        input_df = copy.deepcopy(self.input_df)
        F = self.constants["faradays"]
        z = self.constants["moving_electrons"]
        Kc = fuel_cell_parameters["Kc"]
        alpha = fuel_cell_parameters["alpha"]
        R = self.constants["gas_constant"]
        
        E_OC = []
        i0 = []
        tafel_slope = []
        
        for index, row in input_df.iterrows():
            
            # Current state calculation
            T = row["T"]
            I_fc = row["I_fc"]
            P_fuel = row["P_fuel"]
            P_air = row["P_air"]
            Vfuel = row["Vfuel"]
            Vair = row["Vair"]
            x = row["x"]
            y = row["y"]

            self.current_state = {
                            "I_fc": I_fc,
                            "P_fuel": P_fuel,
                            "P_air": P_air,
                            "T": T,
                            "Vfuel" : Vfuel,
                            "Vair": Vair,
                            "x": x,
                            "y": y
                            }
            
            En = self.calc_en(nom = False)
            
            # Oxygen depletion En 
            if self.current_state["UfO2"]> self.nominal_parameters["ufo2n"]:
            
                En = self.oxygen_depletion(En)
            
          
            E = Kc * En
         
            E_OC.append(E)
            
            A = (R * T)/(z* alpha * F)
            tafel_slope.append(A)
            
            i = self.dynamic_i0()
            i0.append(i)
           
        self.calculated_space["E_OC"] = E_OC 
        self.calculated_space["i0"] = i0
        self.calculated_space["A"] = tafel_slope
        
    #### Dynamic response for input_signal   
    def dynamic_response(self, x0 = 0, transfer_function = "on"):
        """
        

        Parameters
        ----------
        transfer_function : TYPE, optional
            DESCRIPTION. The default is "on".

        Returns
        -------
        None.

        """
        if self.calculated_space.empty:
            
            self.dynamic_block_calculation()
            
        N = self.nominal_parameters["N"]
        I_fc = np.array(self.input_df["I_fc"])
        tafel_slope = np.array(self.calculated_space["A"])
        i0 = self.calculated_space["i0"] 
        Rohm = self.nominal_parameters["Rohm"]
        E_OC =  self.calculated_space["E_OC"]
        response_df = pandas.DataFrame(E_OC, columns = ["E_OC"])
        NA = N * tafel_slope
        
        Td = self.nominal_parameters["Td"]
        num = [1.0]
        den = [Td/3, 1.0]
        
        
        signal_f = NA * np.log(np.divide(I_fc, i0))
        response_df["signal_f"] = pandas.Series(signal_f, dtype = float)
        response_df["t"] = self.input_df["t"]
        response_df["I_fc"] = self.input_df["I_fc"]

        if transfer_function == "off":
            E = E_OC - signal_f
            V_fc = E - Rohm* I_fc
            
        elif transfer_function == "on":
            
            tf = signal.TransferFunction(num, den)
            signal_f_t, signal_f_y, X_out = signal.lsim(tf,
                                                    U =response_df["signal_f"], 
                                                    T =response_df["t"], X0= [x0], interp = False)
            E = E_OC - signal_f_y
            
            V_fc = E - Rohm* I_fc
            response_df["X_out"] = X_out
        response_df["V_fc"] = V_fc
        response_df["E"] = E
        self.response_df = copy.deepcopy(response_df)
        self.add_calc_vector()
    
    #### Input signal generation
    def generate_input_signal(self, i = None,  number_time_steps = None, 
                              time_stamp = None, step_time = None, **kwargs ):
        
        """
        
        i = [i_start, i_end]
        **kwargs = T = None, P_fuel = None, P_air = None,
                                  Vair = None, x = None, y = None,
                                  
        """
        if not isinstance(i, type(None)):
            kwargs["i"] = i
            
        else:
            kwargs["i"] = None
            
            
        kwargs["timestamp"] = time_stamp
        if not isinstance(self.input_df, type(None)):
            
            input_df = copy.deepcopy(self.input_df)
            number_time_steps = len(input_df)
            
        else:
            if isinstance(number_time_steps, type(None)):
                raise ValueError("number_time_steps is required to generate"\
                                 " the signal")
            input_df = pandas.DataFrame()
        
        for key in kwargs.keys():
            
            if key == "i":
                if isinstance(i, type(None)):
                    
                    try:
                        i_signal = input_df["I_fc"]
                        
                        continue
                        
                    except IndexError:
                        
                        raise ValueError("Unable to generate input signal."\
                                         " Provide current start and stop values")
                        
                    
                elif not isinstance(kwargs[key], list) or isinstance(kwargs[key],
                                                                   type(None)):
                    raise ValueError("Start and end values of i should be "\
                                     "provided as list")
                    
                else:
                    
                    if not len(kwargs[key]) == 2:
                        raise ValueError("Only start and end values for "\
                                         "current should be provided as "\
                                        "list and in that order. Check input"\
                                        " values for i")
                            
                    else:
                        
                        element = kwargs[key]
                        i_start = element[0]                
                        i_end = element[1]
                        
                        if i_start >= i_end:
                            raise ValueError("Invalid values for i. Check "\
                                             "input values. Hint: i should be "\
                                            "formatted as i = [i_start, i_stop]")
                                
                        if isinstance(number_time_steps, type(None)):
                            raise ValueError("number_time_steps is required to "\
                                             "generate signal")
                        else:
                            i_signal = np.linspace(i_start, i_end, 
                                                   number_time_steps)
                            
                            input_df["I_fc"] = i_signal
                    
            elif key == "timestamp":
                
                t = []
                
                if not isinstance(input_df["t"], type(None)):
                    number_time_steps = len(input_df["t"])
                    end_time = input_df["t"].iloc[-1]
                    start_time = input_df["t"].iloc[0]
                    
                    duration = (end_time - start_time) / number_time_steps
                    timestamp, _  = create_timestamp(kwargs[key], 
                                                     number_time_steps, duration)
                    input_df["timestamp"] = timestamp
                    continue
                
                else:
                    timestamp, step_duration = create_timestamp(kwargs[key], 
                                                                number_time_steps)
                    
                    total_duration_seconds = step_duration.total_seconds()
                    input_df["timestamp"] = timestamp
    
                    for j in range(0, number_time_steps):
                        
                        if j == 0:
                            time_sec = 0
                            pass
    
                        time_sec = time_sec + total_duration_seconds
                        t.append(time_sec)
    
                    input_df["t"] = t
                
            elif key == "P_fuel":
                
                if isinstance(kwargs[key], type(None)):
                    
                    warnings.warn("P_fuel start and end values are not provided."\
                                  " Taking default values to generate signals."\
                                      " Check results carefully")
                        
                    P_fuel_signal = [self.nominal_parameters["P_fueln"]] * number_time_steps
                    
                else:
                    if len(kwargs[key]) == 1:
                        P_fuel_signal = kwargs[key] * number_time_steps
                        
                    elif not len(kwargs[key]) == 2:
                        
                        raise KeyError("Only start and end values for "\
                                       "P_fuel should be provided as list."\
                                           " Check input values for P_fuel")
                    else:
                        element = kwargs[key]
                        P_fuel_start = element[0]
                        P_fuel_end = element[1]
                        
                        P_fuel_signal = np.linspace(P_fuel_start, P_fuel_end,
                                                    number_time_steps)
                    
                input_df["P_fuel"] = P_fuel_signal
                      
            elif key == "P_air":
                
                if isinstance(kwargs[key], type(None)):
                    warnings.warn("P_air start and end values are not provided."\
                                  " Taking default values to generate signals."\
                                      " Check results carefully")
                    P_air_signal = [self.nominal_parameters["P_airn"]] * number_time_steps
                    
                else:
                    if len(kwargs[key]) == 1:
                        P_air_signal = kwargs[key] * number_time_steps
                        
                    elif not len(kwargs[key]) == 2:
                        
                        raise KeyError("Only start and end values for "\
                                       "P_air should be provided as list."\
                                           " Check input values for P_air")
                    else:
                        element = kwargs[key]
                        P_air_start = element[0]

                        P_air_end = element[1]
                        P_air_signal = np.linspace(P_air_start, P_air_end, 
                                                   number_time_steps)
                    
                input_df["P_air"] = P_air_signal
                
            elif key == 'Vair':
                if isinstance(kwargs[key], type(None)):
                    warnings.warn("Vair start and end values are not provided."\
                                  " Taking default values to generate signals."\
                                      " Check results carefully")
                    Vair_signal = [self.nominal_parameters["Vairn"]] * number_time_steps
                    
                else:
                    
                    if len(kwargs[key]) == 1:
                        Vair_signal = kwargs[key] * number_time_steps
                    elif not len(kwargs[key]) == 2:
                        
                        raise KeyError("Only start and end values for "\
                                       "Vair should be provided as list."\
                                           " Check input values for Vair")
                    else:
                        element = kwargs[key]
                        Vair_start = element[0]

                        Vair_end = element[1]
                        Vair_signal = np.linspace(Vair_start, Vair_end, 
                                                  number_time_steps)
                    
                input_df["Vair"] = Vair_signal
                
            elif key == 'Vfuel':
                
                if isinstance(kwargs[key], type(None)):
                    
                    warnings.warn("Vfuel start and end values are not provided."\
                                  " Taking default values to generate signals."\
                                      " Check results carefully")
                    Vfuel_rate = 2 * self.nominal_parameters["Vairn"]
                    Vfuel_signal = [Vfuel_rate] * number_time_steps
                    
                else:
                    if len(kwargs[key]) == 1:
                        Vfuel_signal = kwargs[key] * number_time_steps
                    elif not len(kwargs[key]) == 2:
                        
                        raise KeyError("Only start and end values for "\
                                       "Vfuel should be provided as list."\
                                           " Check input values for Vfuel")
                    else:
                        element = kwargs[key]
                        Vfuel_start = element[0]

                        Vfuel_end = element[1]
                        Vfuel_signal = np.linspace(Vfuel_start, Vfuel_end, number_time_steps)
                    
                input_df["Vfuel"] = Vfuel_signal
                    
            elif key == 'T':
                
                if isinstance(kwargs[key], type(None)):
                    warnings.warn("T start and end values are not provided."\
                                  " Taking default values to generate signals."\
                                      " Check results carefully")
                    T_signal = [self.nominal_parameters["Tn"]] * number_time_steps
                    
                else:
                    if len(kwargs[key]) == 1:
                        T_signal = kwargs[key] * number_time_steps
                    elif not len(kwargs[key]) == 2:
                        
                        raise KeyError("Only start and end values for "\
                                       "T should be provided as list."\
                                           " Check input values for T")
                    else:
                        element = kwargs[key]
                        T_start = element[0]

                        T_end = element[1]
                        T_signal = np.linspace(T_start, T_end, number_time_steps)
                    
                input_df["T"] = T_signal

            elif key == 'x':
                
                if isinstance(kwargs[key], type(None)):
                    warnings.warn("x start and end values are not provided."\
                                  " Taking default values to generate signals."\
                                      " Check results carefully")
                    x_signal = [self.nominal_parameters["xn"]] * number_time_steps
                    
                else:
                    if len(kwargs[key]) == 1:
                        x_signal = kwargs[key] * number_time_steps
                    elif not len(kwargs[key]) == 2:
                        
                        raise KeyError("Only start and end values for "\
                                       "x should be provided as list."\
                                           " Check input values for x")
                    else:
                        element = kwargs[key]
                        x_start = element[0]

                        x_end = element[1]
      
                        x_signal = np.linspace(x_start, x_end, number_time_steps)
                    
                input_df["x"] = x_signal

            elif key == 'y':
                
                if isinstance(kwargs[key], type(None)):
                    
                    warnings.warn("y start and end values are not provided."\
                                  " Taking default values to generate signals."\
                                      " Check results carefully")
                        
                    y_signal = [self.nominal_parameters["yn"]] * number_time_steps
                    
                else:
                    if len(kwargs[key]) == 1:
                        y_signal = kwargs[key] * number_time_steps
                    elif not len(kwargs[key]) == 2:
                        
                        raise KeyError("Only start and end values for "\
                                       "y should be provided as list."\
                                           " Check input values for y")
                    else:
                        element = kwargs[key]
                        y_start = element[0]

                        y_end = element[1]
                        y_signal = np.linspace(y_start, y_end, number_time_steps)
                    
                input_df["y"] = y_signal

        return input_df
    
    @staticmethod
    def plot_results(x, y, xlabel, ylabel, title):
        """
        

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.
        xlabel : TYPE
            DESCRIPTION.
        ylabel : TYPE
            DESCRIPTION.
        title : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        plot_results(x, y, xlabel, ylabel, title)
    
    def add_calc_vector(self):
        
        self.response_df["En"] = self.En_vector
        self.response_df["PH2"] = self.PH2_vector
        self.response_df["PO2"] = self.PO2_vector
        self.response_df["PH2O"] = self.PH2O_vector
        self.response_df["UfH2"] = self.UfH2_vector
        self.response_df["UfO2"] = self.UfO2_vector
        
    def reset_calc_vectors(self):
        
        self.En_vector = []
        self.PH2_vector = []
        self.PO2_vector = []
        self.UfH2_vector = []
        self.UfO2_vector = []
        self.PH2O_vector = []
        
    def adaptive_control_input(self):
        
        if self.current_state["UfH2"] >1:
            self.current_state["UfH2"] = 0.5
            
        if self.current_state["UfO2"] >1:
            self.current_state["UfO2"] = 0.5

class EquivalentCircuitRL:
    
    def __init__(self, cell_parameters = None, step_signal=None, input_signal=None, test=True):
       
        if isinstance(cell_parameters, type(None)):
            # Nexa 1200
            self.cell_parameters = {"Eoc": 28.32,
                                    "R1" : None,
                                    "R2" : None,
                                    "L": None}
 
        else:
            self.cell_parameters = copy.deepcopy(cell_parameters)

        if test == True:
            
            # Nexa 1200
            self.step_signal ={"I_0" : 13,
                               "I_1" : 33,
                                "t0" : 65.1,
                                "V01": 28.2,
                                "V02": 22.09,
                                "V_inf": 25.12,
                                "Vt1" : 24.14,
                                "t1" : 101.8}
            
        elif not isinstance(step_signal, type(None)):
            self.step_signal = copy.deepcopy(step_signal)
            
        if isinstance(input_signal, type(None)):
            # Create test signal
            t_start = 0
            t_end = 350
            step_start = 65
            step_duration = 285
            signal_amplitude = 33
            signal_name = "I_fc"
            initial_amplitude = 13

            self.input_signal = create_signal(t_start, t_end, step_start, 
                                              step_duration, signal_amplitude, 
                                              signal_name, initial_amplitude, 
                                              plot = False)
            
            
        else:
            self.input_signal = copy.deepcopy(input_signal)
            
            
        self.cell_state = {"I": None,
                           "V": None,
                           "t": None}
        
        self.output_signal = pandas.DataFrame()
            
    def calculate_r(self):
        """
        

        Returns
        -------
        None.

        """
        del_V1, del_V2 = self.delta_voltage()
        del_I = self.step_signal["I_1"] - self.step_signal["I_0"]
        R2 = del_V2/ del_I
        R1 = del_V1/del_I - R2
        
        self.cell_parameters["R1"] = R1
        self.cell_parameters["R2"] = R2
        
        
    def cell_voltage(self, t = None, t0 = None, I_0 = None, I_1 = None, I_fc = None, mode = "static"):
        """
        

        Parameters
        ----------
        t : TYPE, optional
            DESCRIPTION. The default is None.
        t0 : TYPE, optional
            DESCRIPTION. The default is None.
        I_0 : TYPE, optional
            DESCRIPTION. The default is None.
        I_1 : TYPE, optional
            DESCRIPTION. The default is None.
        I_fc : TYPE, optional
            DESCRIPTION. The default is None.
        mode : TYPE, optional
            DESCRIPTION. The default is "static".

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        V_fc : TYPE
            DESCRIPTION.

        """        
        R2 = self.cell_parameters["R2"]
        Eoc = self.cell_parameters["Eoc"]
        
        if mode == "static":

            V_fc = Eoc - R2 * I_fc
            
        elif mode == "dynamic":

            I_L = self.inductor_current(t, t0, I_0, I_1)
            R1 = self.cell_parameters["R1"]
            V_fc = Eoc - R2 * I_fc - R1 * (I_fc - I_L)
            
        else:
            raise ValueError("Invalid mode = {}".format(mode))
        
        return V_fc
    
    def inductor_current(self, t, t0 = None, I_0 = None, I_1 = None):
        """
        

        Parameters
        ----------
        t : TYPE
            DESCRIPTION.
        t0 : TYPE, optional
            DESCRIPTION. The default is None.
        I_0 : TYPE, optional
            DESCRIPTION. The default is None.
        I_1 : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        I_L : TYPE
            DESCRIPTION.

        """
        if isinstance(I_0, type(None)):
            I_0 = self.step_signal["I_0"]
            
        if isinstance(t0, type(None)):
            t0 = self.step_signal["t0"]
        
        if isinstance(I_0, type(None)):
            I_1 = self.step_signal["I_1"]
            
        I_1 = self.step_signal["I_1"]
        R1 = self.cell_parameters["R1"]
        L = self.cell_parameters["L"]
        tc_L = L/R1
        I_L = I_1 + (I_0 - I_1) * np.exp( (-1) * (t - t0)/ tc_L)
     
        return I_L

    def calc_voltage_t(self, t):
        """
        

        Parameters
        ----------
        t : TYPE
            DESCRIPTION.

        Returns
        -------
        Vt : TYPE
            DESCRIPTION.

        """
        I_0 = self.step_signal["I_0"]
        R2 = self.cell_parameters["R2"]
        Eoc = self.cell_parameters["Eoc"]
        I_1 = self.step_signal["I_1"]
        R1 = self.cell_parameters["R1"]
        L = self.cell_parameters["L"]
        t0 = self.step_signal["t0"]
        
        V_inf = Eoc - R2 * I_1
        self.step_signal["V_inf"] = V_inf
        
        if isinstance(R1, type(None)) or isinstance(R2, type(None)):
            self.calculate_r()
            R1 = self.cell_parameters["R1"]
            R2 = self.cell_parameters["R2"]
        
        if isinstance(self.cell_parameters["L"], type(None)):
            self.calc_inductance()
            L = self.cell_parameters["L"]
            
        tc_L = L/R1
        Vt = V_inf - R1* (I_1 - I_0)* np.exp( (-1) * (t - t0)/ tc_L)
        
        return Vt

    def delta_voltage(self):
        """
        

        Returns
        -------
        del_V1 : TYPE
            DESCRIPTION.
        del_V2 : TYPE
            DESCRIPTION.

        """
        V01 = self.step_signal["V01"]
        V02 = self.step_signal["V02"]
        V_inf = self.step_signal["V_inf"]
        del_V1 = V01 - V02
        del_V2 = V01 - V_inf
        
        return del_V1, del_V2
    
    def calc_inductance(self):
        """
        

        Returns
        -------
        None.

        """
        del_V1, del_V2 = self.delta_voltage()
        R1 = self.cell_parameters["R1"]
        V_inf = self.step_signal["V_inf"]
        Vt1 = self.step_signal["Vt1"]
        del_V3 = V_inf - Vt1
        t0 = self.step_signal["t0"]
        t1 = self.step_signal["t1"]
        
        L = ((t1 - t0) * R1)/ (np.log((del_V1 - del_V2)/del_V3))
        self.cell_parameters["L"] = L
    
    def calc_cell_parameters(self):
        """
        

        Returns
        -------
        None.

        """
        self.calculate_r()
        self.calc_inductance()
        
    def output_voltage(self):
        """
        

        Returns
        -------
        V_fc_df : TYPE
            DESCRIPTION.

        """
        input_df = copy.deepcopy(self.input_signal)
        
        if not bool(self.cell_parameters.get('L')):
            self.calc_cell_parameters()
        
        dynamic_index = []
        V_fc_df = pandas.DataFrame()
        
        for index, row in input_df.iterrows():
            
            V_fc = []
            
            if index in dynamic_index:
                
                continue
            
            elif index == len(input_df) - 1:
                I_fc = row["I_fc"]
                voltage = self.cell_voltage(I_fc=I_fc, mode="static")
                V_fc.append(voltage)
                voltage_df = pandas.DataFrame(V_fc, columns = ["V_fc"])
                
            else:
                
                I_0 = row["I_fc"]
                I_1 = input_df.at[index+1, "I_fc"]
                
                if I_1 > I_0:
                    
                    input_df_size = input_df.groupby("I_fc").size()
                    I_1_size = input_df_size[I_1]
                    
                    dynamic_index = [i for i in range(index, index + I_1_size)]
                    
                    
                    for j in dynamic_index:
                        
                        t0 = row["t"]
                        t = input_df.at[j + 1, "t"]
                        i0 = row["I_fc"]
                        i1 = input_df.at[j + 1, "I_fc"]
    
                        voltage = self.cell_voltage(t=t, t0=t0, I_0=i0, I_1=i1,
                                                    I_fc=i1, mode="dynamic")
                        
                        V_fc.append(voltage)
                        
                    voltage_df = pandas.DataFrame(V_fc, columns = ["V_fc"])
                
                else:
                    voltage = self.cell_voltage(I_fc = I_0)
                    V_fc.append(voltage)
                    voltage_df = pandas.DataFrame(V_fc, columns = ["V_fc"])
                    
            V_fc_df = pandas.concat([V_fc_df, voltage_df], axis = 0, 
                                    ignore_index=True)
            
        self.output_signal["I_fc"] = self.input_signal["I_fc"]
        self.output_signal["t"] = self.input_signal["t"]
        self.output_signal["V_fc"] = V_fc_df
        
        return V_fc_df
    
    @staticmethod
    def plot_results(x, y, xlabel, ylabel, title):
        """
        

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.
        xlabel : TYPE
            DESCRIPTION.
        ylabel : TYPE
            DESCRIPTION.
        title : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        plot_results(x, y, xlabel, ylabel, title)
        
   
class SteadyStateEmprical:
    
    def __init__(self, cell_parameters=None, input_df=None):
        
        if isinstance(cell_parameters, type(None)):
            self.cell_parameters = {"N" : 64,
                                    "c1" : -0.8535,
                                    "c2" : 2.4316 * 10**(-3),
                                    "c3" : 3.7545 * 10**(-5),
                                    "c4" : -9.54 * 10**(-5),
                                    "Rc" : 0.1 * 10**(-3),
                                    "lambd" : 13.082,
                                    "B" : 0.0136,
                                    "M_a" : 240,
                                    "l" : 0.0178,
                                    "Imax" : 225}
        else:
            self.cell_parameters = cell_parameters

        self.input_df = copy.deepcopy(input_df)
   
    def stack_voltage(self, Ifc, Tfc, PH2, PO2):
        """
        

        Parameters
        ----------
        Ifc : TYPE
            DESCRIPTION.
        Tfc : TYPE
            DESCRIPTION.
        PH2 : TYPE
            DESCRIPTION.
        PO2 : TYPE
            DESCRIPTION.

        Returns
        -------
        Vstack : TYPE
            DESCRIPTION.

        """
        N = self.cell_parameters["N"]
        Vact = self.activation_voltage(Ifc, Tfc, PO2)
        Vohm = self.ohmic_voltage_drop(Ifc, Tfc)
        Vcon = self.concentration_voltage_drop(Ifc)
        En = self.calc_nernst(Tfc, PH2, PO2)
        Vstack = N * (En - Vact - Vohm - Vcon)
        return Vstack
    
    def activation_voltage(self, Ifc, Tfc, PO2):
        """
        

        Parameters
        ----------
        Ifc : TYPE
            DESCRIPTION.
        Tfc : TYPE
            DESCRIPTION.
        PO2 : TYPE
            DESCRIPTION.

        Returns
        -------
        Vact : TYPE
            DESCRIPTION.

        """
        cell_parameters = copy.deepcopy(self.cell_parameters)
        c1 = cell_parameters["c1"]
        c2 = cell_parameters["c2"]
        c3 = cell_parameters["c3"]
        c4 = cell_parameters["c4"]

        Co2 = self.calculate_co2(Tfc, PO2)

        Vact = (-1)*(c1 + c2 *Tfc + c3*Tfc* np.log(Co2) + c4*Tfc*np.log(Ifc))
        
        return Vact
    
    def ohmic_voltage_drop(self, Ifc, Tfc):
        """
        

        Parameters
        ----------
        Ifc : TYPE
            DESCRIPTION.
        Tfc : TYPE
            DESCRIPTION.

        Returns
        -------
        Vohm : TYPE
            DESCRIPTION.

        """
        cell_parameters = copy.deepcopy(self.cell_parameters)
        M_a = cell_parameters["M_a"]
        lambd = cell_parameters["lambd"]
        rho_m = 181.6*(1 + 0.03*(Ifc/M_a) + 0.062 * ((Tfc/303) ** (2))* ((Ifc/M_a)** (2.5)))/((lambd - 0.634 - 3 *(Ifc/M_a))*np.exp(4.18* ((Tfc - 303)/Tfc)))
        
        l = cell_parameters["l"]
        Rc = cell_parameters["Rc"]
        Rm = (rho_m *l) / M_a
       
        Vohm = Ifc*(Rm + Rc)
        
        return Vohm
    
    def concentration_voltage_drop(self, Ifc):
        """
        

        Parameters
        ----------
        Ifc : TYPE
            DESCRIPTION.

        Returns
        -------
        Vcon : TYPE
            DESCRIPTION.

        """
        B = self.cell_parameters["B"]
        # ratio of current density and max current density of a cell is the same as ratio of input current and max current
        Imax = self.cell_parameters["Imax"]
        density_ratio = Ifc/Imax
        Vcon = (-B) * np.log(1- density_ratio)
        
        return Vcon
    
    def calc_nernst(self, Tfc, PH2, PO2):
        """
        

        Parameters
        ----------
        Tfc : TYPE
            DESCRIPTION.
        PH2 : TYPE
            DESCRIPTION.
        PO2 : TYPE
            DESCRIPTION.

        Returns
        -------
        En : TYPE
            DESCRIPTION.

        """
        En = 1.229 - 0.85 * (10**(-3))*(Tfc -298.15)+ 4.3085*(10**(-5))*Tfc*np.log(PH2*(PO2 ** (0.5)))
        
        return En
    
    def calculate_co2(self, Tfc, PO2):
        """
        

        Parameters
        ----------
        Tfc : TYPE
            DESCRIPTION.
        PO2 : TYPE
            DESCRIPTION.

        Returns
        -------
        Co2 : TYPE
            DESCRIPTION.

        """
        Co2 = (PO2 *np.exp(498/Tfc))/(5.08 * (10**(6)))
        return Co2
    
    def generate_input_signal(self, I=None, T=None, PH2=None, PO2=None, num=None):
        """
        

        Parameters
        ----------
        I : TYPE, optional
            DESCRIPTION. The default is None.
        T : TYPE, optional
            DESCRIPTION. The default is None.
        PH2 : TYPE, optional
            DESCRIPTION. The default is None.
        PO2 : TYPE, optional
            DESCRIPTION. The default is None.
        num : TYPE, optional
            DESCRIPTION. The default is None.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        input_signal : TYPE
            DESCRIPTION.

        """
        input_signal = pandas.DataFrame()
        if not isinstance(I, type(None)):
            if not isinstance(I, list):
                raise ValueError("The start and end values for I should be provided as a list")
            
            else:
                if not num:
                    num = 1000
                
                if len(I)==1:
                    Ifc = num*I
                    
                elif not len(I)==2:
                    raise ValueError("only start and end values should be provided")
               
                else:
                    start = I[0]
                    end = I[1]
                    
                    Ifc = np.linspace(start, end, num)
                input_signal["Ifc"] = Ifc
                
        else:
            
            Ifc = np.linspace(0, 100, num)
            input_signal["Ifc"] = Ifc
            
                
        if not isinstance(T, type(None)):
            if not isinstance(T, list):
                raise ValueError("The start and end values for T should be provided as a list")
            
            else:
                if not num:
                    num = 1000
                    
                if len(T)==1:
                    Tfc = num*T  
                    
                elif not len(T)==2:
                    raise ValueError("only start and end values should be provided")
                  
                else:
                    start = T[0]
                    end = T[1]
                    
                    Tfc = np.linspace(start, end, num)
                input_signal["Tfc"] = Tfc
                
        else:
            
            Tfc = np.linspace(273, 373, num)
            input_signal["Tfc"] = Tfc
            
        if not isinstance(PH2, type(None)):
            if not isinstance(PH2, list):
                raise ValueError("The start and end values for PH2 should be provided as a list")
            
            else:
                if not num:
                    num = 1000
                
                if len(PH2)==1:
                    PH2_fc = num*PH2
                    
                elif not len(PH2)==2:
                    raise ValueError("only start and end values should be provided")
                   
                else:
                    start = PH2[0]
                    end = PH2[1]
                    
                    PH2_fc = np.linspace(start, end, num)
                input_signal["PH2"] = PH2_fc
                
        else:
            
            PH2_fc = np.linspace(0.5, 1, num)
            input_signal["PH2"] = PH2_fc
            
        if not isinstance(PO2, type(None)):
            if not isinstance(PO2, list):
                raise ValueError("The start and end values for PH2 should be provided as a list")
            
            else:
                if not num:
                    num = 1000
                
                if len(PO2)==1:
                    PO2_fc = num*PO2
                    
                elif not len(PO2)==2:
                    raise ValueError("only start and end values should be provided")
                    
                else:  
                    start = PO2[0]
                    end = PO2[1]
                    
                    PO2_fc = np.linspace(start, end, num)
                input_signal["PO2"] = PO2_fc
                
        else:
            
            PO2_fc = np.linspace(0.5, 1, num)
            input_signal["PO2"] = PO2_fc
            
        self.input_df = copy.deepcopy(input_signal)
        
        return input_signal
                
    def run_steady_state(self):
        """
        

        Returns
        -------
        response_df : TYPE
            DESCRIPTION.

        """
        input_df = copy.deepcopy(self.input_df)
        response_df = pandas.DataFrame()
        Vstack = []
        if isinstance(input_df, type(None)):
            input_df = self.generate_input_signal()
            
        for index, row in input_df.iterrows():
            Ifc = row["Ifc"]
            Tfc = row["Tfc"]
            PH2 = row["PH2"]
            PO2 = row["PO2"]
            
            V = self.stack_voltage(Ifc, Tfc, PH2, PO2)
            Vstack.append(V)
            
        response_df["Vfc"] = Vstack
        return response_df


class StateSpaceModel:
    
    def __init__(self, input_df = None, current_profile = None):
        """
        

        Parameters
        ----------
        input_df : TYPE, optional
            DESCRIPTION. The default is None.
        current_profile : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        # cell parameters for avista fuel cell
        self.cell_parameters = {"A_s": 3.2 * 10**(-2),
                                "a": -3.08 * 10**(-3),
                                "a0": 1.3697,
                                "b": 9.724 * 10*(-5),
                                "Cfc":500,
                                "C": 10,
                                "E0_cell": 1.23,
                                "e": 2,
                                "F": 96487,
                                "del_G": 237.2 * 10**(3),
                                "hs": 37.5,
                                "K_I": 1.87*10**(-3),
                                "K_T": -2.37 * 10**(-3),
                                "M_fc": 44,
                                "mH2O": 8.614 * 10**(-5),
                                "ns": 48,
                                "PH2O": 2,
                                "R": 8.31,
                                "Roc": 0.28,
                                "Va": 10**(-3),
                                "Vc": 10**(-3),
                                "lamb_a": 60,
                                "lamb_c": 60,
                                "PH2O_e": 1
                                }
        
        self.current_state_variables = {
                                "mO2_net" :None,
                                "mH2_net" :  None,
                                "mH2O_net" : None,
                                "T": None,
                                "PH2": None,
                                "PO2": None,
                                "PH2O": None,
                                "Qc": None,
                                "Qe": None,
                                "Ql": None,
                                "Vc" : None
                                }
        self.current_theta = None
        
        self.input_states = {   "I": None,
                                "u_Pa": None,
                                "u_Pc": None,
                                "u_Tr": None
                               }
        
        if isinstance(input_df, type(None)):
            self.input_df = pandas.DataFrame()
            
        else:
            self.input_df = copy.deepcopy(input_df)
            
            
        self.output_df = pandas.DataFrame()
        
        if isinstance(current_profile, type(None)):
            self.input_signal = dum_input_current()
            
        self.counter = 0
        self.x = None

    #### Matrix elements calculation
    @property
    def matrix_a_44(self):
        """
        

        Returns
        -------
        a_44 : TYPE
            DESCRIPTION.

        """
        cell_parameters = copy.deepcopy(self.cell_parameters)
        hs = cell_parameters["hs"]
        ns = cell_parameters["ns"]
        A_s = cell_parameters["A_s"]
        M_fc = cell_parameters["M_fc"]
        Cfc = cell_parameters["Cfc"]
        
        a_44 = (-hs *ns * A_s)/(M_fc * Cfc)
        return a_44
    
    def matrix_a_11(self):
        """
        

        Returns
        -------
        a_11 : TYPE
            DESCRIPTION.

        """
        cell_parameters = copy.deepcopy(self.cell_parameters)
        I = self.input_states["I"]
        
        C = cell_parameters["C"]
        Vact = self.activation_loss()
        Rconc = self.concentration_resistance()
        
        Ract = Vact/I
        
        a_11 = (-1)/ (C*(Ract + Rconc))
        
        
        return a_11
    
    #### Polarisation loss 
    def activation_loss(self):
        """
        

        Returns
        -------
        Vact : TYPE
            DESCRIPTION.

        """
        
        cell_parameters = copy.deepcopy(self.cell_parameters)
        I = self.input_states["I"]
        a = cell_parameters["a"]
        b = cell_parameters["b"]
        a0 = cell_parameters["a0"]
        T = self.input_states["u_Tr"]
        Vact = a0 + T * (a + (b*np.log(I)))
        
        return Vact
    
    # Valid only for Avista labs SR-12
    def concentration_resistance(self):
        """
        

        Returns
        -------
        Rconc : TYPE
            DESCRIPTION.

        """
        I = self.input_states["I"]
        Rconc0 = 0.080312
        Rconc1 = 5.2211*(10**(-8))*(I**6) - 3.4578*(10**(-6))*(I**5) + 8.6437*(10**(-5))*(I**4) - 0.010089**(I**3) + 0.005554*(I**2) - 0.010542*I
        T = self.input_states["u_Tr"]
        Rconc2 = 0.0002747*(T-298)
        Rconc = Rconc0 + Rconc1 + Rconc2
        
        return Rconc
    
    def ohmic_loss(self):
        """
        

        Returns
        -------
        Ro : TYPE
            DESCRIPTION.

        """
        I = self.input_states["I"]
        T = self.input_states["u_Tr"]
        cell_parameters = copy.deepcopy(self.cell_parameters)
        Roc = cell_parameters["Roc"]
        K_I = cell_parameters["K_I"]
        K_T = cell_parameters["K_T"]
        Ro = Roc + K_I*I + K_T * T
        
        return Ro
    #### Matrix calculation
    def calc_theta(self):
        """
        

        Returns
        -------
        None.

        """
        I = self.input_states["I"]
        current_state_variables = copy.deepcopy(self.current_state_variables)
        cell_parameters = copy.deepcopy(self.cell_parameters)
        R = cell_parameters["R"]
        Va = cell_parameters["Va"]
        mH2Oa_in = cell_parameters["mH2O"]
        PH2Oa_in = cell_parameters["PH2O_e"]
        F = cell_parameters["F"]
        ns = cell_parameters["ns"]
        del_G0 = cell_parameters["del_G"]
        PH2O_in = cell_parameters["PH2O"]
        x4 = current_state_variables["T"]
        x5 = current_state_variables["PH2"]
        x6 = current_state_variables["PO2"]
        x7 = current_state_variables["PH2O"]
        
        theta_1 = (R*mH2Oa_in*x4)/(Va*PH2Oa_in)
        theta_2 = (R*x4)/(2*Va*F)
        
        Vc = cell_parameters["Vc"]
        mH2Oc_in = cell_parameters["mH2O"]
        PH2Oc_in = cell_parameters["PH2O_e"]
        theta_3 = (R*mH2Oc_in*x4)/(Vc*PH2Oc_in)
        theta_4 = (R*x4)/(4*Vc*F)
        
        x7 = current_state_variables["PH2O"]
        theta_5 = (R* mH2Oc_in *(PH2O_in - x7))/(Vc*(PH2Oc_in))
        theta_6 = (ns*del_G0)/(2*F) - ((ns*R*x4)/(2*F))*np.log((x5*(x6**0.5))/x7)
        
        Rconc = self.concentration_resistance()
        Ro = self.ohmic_loss()
        Vo = Ro * I
        Vconc = Rconc*I
        Vact = self.activation_loss()
        E0_cell = cell_parameters["E0_cell"]
        
        theta_7 = ns*(E0_cell + (R*x4)/(2*F) * np.log(x5*(x6**0.5)/x7) - Vact - Vconc - Vo)
        Mfc = cell_parameters["M_fc"]
        Cfc = cell_parameters["Cfc"]
        theta_8 = ns*((2*E0_cell/Mfc*Cfc) + (R*x4/F*Mfc*Cfc)*np.log(x5*(x6**0.5)/x7) - Vact - Vconc - Vo)
        self.current_theta = {"theta_1": theta_1,
                      "theta_2": theta_2,
                      "theta_3": theta_3,
                      "theta_4": theta_4,
                      "theta_5": theta_5,
                      "theta_6": theta_6,
                      "theta_7": theta_7,
                      "theta_8": theta_8}   
        
        
    def matrix_a(self):
        """
        

        Returns
        -------
        a_matrix : TYPE
            DESCRIPTION.

        """
        I = self.input_states["I"]
        cell_parameters = copy.deepcopy(self.cell_parameters)
        hs = cell_parameters["hs"]
        ns = cell_parameters["ns"]
        As = cell_parameters["A_s"]
        a_matrix = np.zeros((11,11))
        a_44 = self.matrix_a_44
        a_11 = self.matrix_a_11()
        cell_parameters = copy.deepcopy(self.cell_parameters)
        self.calc_theta()
        theta = copy.deepcopy(self.current_theta)
        
        theta_1 = theta["theta_1"]
        theta_5 =  theta["theta_2"]
        theta_3 = theta["theta_3"]
        lamb_a = cell_parameters["lamb_a"]
        lamb_c = cell_parameters["lamb_c"]
      
        for index_i, row in enumerate(a_matrix):
            for index_j, element in enumerate(row):
                if index_i == index_j:
                    if index_j == 0 or index_j == 2:
                        a_matrix[index_i][index_j] = (-1/lamb_c)
                        
                    elif index_i == 1:
                        a_matrix[index_i][index_j] = (-1/lamb_a)
                        
                    elif index_i == 3:
                        a_matrix[index_i][index_j] = a_44
                        
                    elif index_i == 4:
                        a_matrix[index_i][index_j] = (-2) * theta_1
                        
                    elif index_i == 5:
                        a_matrix[index_i][index_j] = (-2) * theta_3
                        
                    elif index_i ==10:
                        a_matrix[index_i][index_j] = a_11
                        
                elif index_i == 6 and index_j ==3:
                    a_matrix[index_i][index_j] = (-2) * theta_5
                    
                elif index_i == 9 and index_j == 3:
                    a_matrix[index_i][index_j] = hs * ns * As
                    
        return a_matrix
        
        
    def matrix_b(self):
        """
        

        Returns
        -------
        b_matrix : TYPE
            DESCRIPTION.

        """
        
        b_matrix = np.zeros((11,3))
        cell_parameters = copy.deepcopy(self.cell_parameters)
        
        b_12 = -1 * self.matrix_a_44
        self.calc_theta()
        theta = copy.deepcopy(self.current_theta)
        theta_1 = theta["theta_1"]
        theta_3 = theta["theta_3"]
        hs = cell_parameters["hs"]
        ns = cell_parameters["ns"]
        As = cell_parameters["A_s"]
        
        b_matrix[1][2] = b_12
        b_matrix[2][0] = 2*theta_1
        b_matrix[3][1] = 2*theta_3
        b_matrix[9][2] = -1*(hs * ns * As)
        # for index_i, row in b_matrix:
        #     for index_j, element in row:
        #         if index_i == 1 and index_j ==2:
        #             b_matrix[index_i][index_j] = b_12
                    
        #         elif 
        return b_matrix
        
    def matrix_g(self):
        """
        

        Returns
        -------
        g_matrix : TYPE
            DESCRIPTION.

        """
        cell_parameters = copy.deepcopy(self.cell_parameters)
        g_matrix = np.zeros(11)
        lamb_c = cell_parameters["lamb_c"]
        lamb_a = cell_parameters["lamb_a"]
        F = cell_parameters["F"]
        C = cell_parameters["C"]
        
        current_theta = copy.deepcopy(self.current_theta)
        theta_2 = current_theta["theta_2"]
        theta_4 = current_theta["theta_4"]
        theta_6 = current_theta["theta_6"]
        theta_7 = current_theta["theta_7"]
        theta_8 = current_theta["theta_8"]
        
        g_matrix[0] = 1/(4*lamb_c*F)
        g_matrix[1] = 1/(2*lamb_a*F)
        g_matrix[2] = 1/(2*lamb_c*F)
        g_matrix[3] = -theta_8
        g_matrix[4] = -theta_2
        g_matrix[5] = -theta_4
        g_matrix[6] = 2*theta_4
        g_matrix[7] = theta_6
        g_matrix[8] = theta_7
        g_matrix[10] = 1/C
        
        g_matrix_T = np.transpose(g_matrix)
        return g_matrix_T
    
    
    def model(self, t, x):
        """
        

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        u : TYPE
            DESCRIPTION.

        Returns
        -------
        dx_dt : TYPE
            DESCRIPTION.

        """
        u = np.array(self.control_input_function(t))
        self.x = copy.deepcopy(x)
        
        
        self.current_state_variables = {
                                "mO2_net" :x[0],
                                "mH2_net" :  x[1],
                                "mH2O_net" : x[2],
                                "T": x[3],
                                "PH2": x[4],
                                "PO2": x[5],
                                "PH2O": x[6],
                                "Qc": x[7],
                                "Qe": x[8],
                                "Ql": x[9],
                                "Vc" : x[10]
                                }
        a_matrix = self.matrix_a()
        
        b_matrix = self.matrix_b()
        g_matrix_T = self.matrix_g()
        
        dx_dt = a_matrix@x + (b_matrix @ u).flatten() + g_matrix_T
        
        
        return dx_dt
    
    def control_input_function(self,t):
        """
        

        Parameters
        ----------
        t : TYPE
            DESCRIPTION.

        Returns
        -------
        list
            DESCRIPTION.

        """
        input_df = copy.deepcopy(self.input_df)
        
        u_Pa = input_df.loc[self.counter, "u_Pa"]
        u_Pc = input_df.loc[self.counter, "u_Pc"]
        u_Tr = input_df.loc[self.counter, "u_Tr"]
        I = input_df.loc[self.counter, "I"]
        #self.counter = self.counter+1
        self.input_states["I"] = I
        
        self.input_states["u_Tr"] = u_Tr
        self.input_states["u_Pa"] = u_Pa
        self.input_states["u_Pc"] = u_Pc
        
        return [u_Pa, u_Pc, u_Tr]
    
    
    #### Solver and Simulation
    def ode_solver(self, initial_state_vector = None, solver = "LSODA"):
        """
        

        Parameters
        ----------
        initial_state_vector : TYPE, optional
            DESCRIPTION. The default is None.
        input_df : TYPE, optional
            DESCRIPTION. The default is None.
        solver : TYPE, optional
            DESCRIPTION. The default is "odeint".

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if self.input_df.empty:
            warnings.warn("Input_df is not provided. Generating Default "\
                          "profile based input_signal. Use {} to generate "\
                              "input signal".format(self.generate_input_signal.__name__))
            self.generate_input_signal()
        
        if isinstance(initial_state_vector, type(None)):
            x0 = {
                    "mO2_net" :8.614 * 10**(-5),
                    "mH2_net" :  8.614 * 10**(-5),
                    "mH2O_net" : 8.614 * 10**(-5),
                    "T": 308,
                    "PH2": 5,
                    "PO2": 5,
                    "PH2O": 5,
                    "Qc": 1.5,
                    "Qe": 1.2,
                    "Ql": 1,
                    "Vc" : 0
                    }
        
        
        x0_list = list(x0.values())
        
        #x0_list_T = np.transpose(x0_list)
        t = np.array(self.input_df["t"])
        t_span = (t[0], t[-1])
        if solver=="LSODA":
            x = solve_ivp(self.model, t_span, x0_list, method='LSODA')
            
        elif solver=="odeint":
            x = odeint(self.model, x0_list, t)
            
        #y = self.current_theta["theta_7"]
        return x
            
        
        
        # if not isinstance(input_df, type(None)):
        #     if not isinstance(self.input_df, type(None)):
        #         warnings.warn('Overwriting input_df that was defined during object initialisation')
        #         self.input_df = copy.deepcopy(input_df)
                
        #     else:
        #         self.input_df = copy.deepcopy(input_df)
                
        # if isinstance(self.input_df, type(None)):
        #     raise ValueError('Input operating condition for simulation')
            
            
        # if isinstance(initial_state_df, type(None)) or len(input_df) != len(initial_state_df):
        #     warnings.warn(' Poorly initialised state vectors. Reverting to'\
        #                   ' default initial state vector profile. Check results carefully')
                
            
        
        # for (index, row1), (index2, row2) in zip(input_df.iterrows(), initial_state_df):
        #     input_vector = self.add_current_profile(row1)
        #     self.input_states = copy.deepcopy(input_vector)
        #     t = input_vector["t"]
            
        #     if solver == "odeint" :
                
        #         x = odeint(self.model, x0, t)
                
    
    #### Input signal generation
    def generate_input_signal(self, I = None, profile = "Default", **kwargs):
        """
        I = [start, stop]
        **kwargs = u_pa, u_pc, u_tr
        profile = "Default", "step", "linear"

        Parameters
        ----------
        I : TYPE, optional
            DESCRIPTION. The default is None.
        profile : TYPE, optional
            DESCRIPTION. The default is "Default".
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if profile == "Default":
            if not isinstance(I, type(None)):
                raise ValueError("Custom value of I is not allowed in Default "\
                                 "profile. Choose another profile to vary I")
            input_df = copy.deepcopy(self.input_signal)
            l = len(input_df)
            try:
                if len(kwargs["u_Pa"]) ==1:
                    u_Pa = l*[kwargs["u_Pa"]]
                    
                else:
                    raise ValueError("u_Pa signal cannot be varied in Default"\
                                     "mode. Choose profile as linear or step to modify value")
                
            except KeyError:
                u_Pa = l*[5]
                
            try:
                if len(kwargs["u_Pc"]) ==1:
                    u_Pc = l*[kwargs["u_Pc"]]
                    
                else:
                    raise ValueError("u_Pc signal cannot be varied in Default"\
                                     "mode. Choose profile as linear or step to modify value")
                
            except KeyError:
                u_Pc = l*[5]
                
            
            try:
                if len(kwargs["u_Tr"]) ==1:
                    u_Tr = l*[kwargs["u_Tr"]]
                    
                else:
                    raise ValueError("u_Tr signal cannot be varied in Default"\
                                     "mode. Choose profile as linear or step to modify value")
                
            except KeyError:
                u_Tr = l*[308]
            
            input_df["u_Pa"] = u_Pa
            input_df["u_Pc"] = u_Pc
            input_df["u_Tr"] = u_Tr
              
        else:
            # more signal types in future
            pass
            
        self.input_df = copy.deepcopy(input_df)
    
        
    
    def reset_states(self):
        
        self.current_state_variables = {
                                "mO2_net" :None,
                                "mH2_net" :  None,
                                "mH2O_net" : None,
                                "T": None,
                                "PH2": None,
                                "PO2": None,
                                "PH2O": None,
                                "Qc": None,
                                "Qe": None,
                                "Ql": None,
                                "Vc" : None
                                }
        self.current_theta = None

             
#%% Data Class  
            
class PreprocessData:
    def __init__(self, fuelcell_model = None, data = None, **kwargs):
        
        self.model_fc = fuelcell_model
        self.scaler = None
        self.tensor_data = None
        # self.is_scaled = None
        self.init_args = copy.deepcopy(kwargs)
        self.data_simulated = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train_scaled = None
        self.y_test_scaled = None
        self.column_number = None
        self.scaled_data = None
        if isinstance(data, type(None)):
            self.simulate_model()
        else:
            self.data_simulated = copy.deepcopy(data)
        
    def simulate_model(self):
        """
        

        Raises
        ------
        ValueError
            DESCRIPTION.
        KeyError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        fc_model = self.model_fc
        
        if isinstance(fc_model, type(None)):
            raise ValueError("Provide fuel cell model while Class instance initiation")
            
        # in progress
        elif fc_model == "ElectroChemicalDynamic":
            try:
                electro_model = self.init_args["electro_model"]
                test_mode = self.init_args["test_mode"]
                input_vector = self.init_args["input_vector"]
                
            except KeyError:
                raise KeyError("Initialisation arguments for Fuel cell model should be provided as in kwargs")
                
            sim_model = ElectroChemicalDynamic(electro_model, test_mode, input_vector)
            data = sim_model.select_model()
            self.data_simulated = copy.deepcopy(data)
            
        # in progress
        elif fc_model == "CorreaModel":
            
            try:
                current_lower = self.init_args["current_lower"]
                current_upper = self.init_args["current_upper"]
        
            
            except KeyError:
                raise KeyError("Initialisation arguments for Fuel cell model should be provided as in kwargs")
                
                
            sim_model = CorreaModel(current_lower, current_upper)
        
        elif fc_model == "GenericMatlabModel":
            try:
                nominal_parameters = self.init_args["nominal_paramaters"]
                input_df = self.init_args["input_df"]
                sim_model = GenericMatlabModel(nominal_parameters, input_df)
                
            except KeyError:
                sim_model = GenericMatlabModel()
            
            try:
                x0 = self.init_args("x0")
                
            except KeyError:
                x0 = 0
            sim_model.dynamic_response(x0)
            V_fc = sim_model.response_df["V_fc"].tolist()
            data = sim_model.input_df
            data["V_fc"] = V_fc
            self.data_simulated = copy.deepcopy(data)
            
        elif fc_model == "EquivalentCircuitRL":
            try:
                cell_parameters = self.init_args["cell_parameters"]
                step_signal = self.init_args["step_signal"]
                input_signal = self.init_args["input_signal"]
                test = self.init_args["test"]
                sim_model = EquivalentCircuitRL(cell_parameters, step_signal, input_signal, test)
                
            except KeyError:
                sim_model = EquivalentCircuitRL()

            V_fc = sim_model.output_voltage()
            data = copy.deepcopy(sim_model.output_signal)
            
        elif fc_model=='GenericMatlabModel_aug':
            P_fuel = [1.5, 3]
            P_air = [1, 3]
            temperature = [333, 373]
            pfuel = np.linspace(P_fuel[0], P_fuel[1], 10)
            pair = np.linspace(P_air[0], P_air[1], 10)
            # vair = np.linspace(Vair[0], Vair[1], 10)
            # vfuel = np.linspace(Vfuel[0], Vfuel[1], 10)
            T = np.linspace(temperature[0], temperature[1], 10)
            I_fc = np.linspace(0.01,200, 120)
            t = np.linspace(0, 500, 120)
            input_df = pandas.DataFrame()
            input_df["I_fc"] = I_fc
            input_df["t"] = t
            df = pandas.DataFrame()

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
            for indexi, i in enumerate(T):
                for indexj , j in enumerate(pair):
                    
                    for indexz, z in enumerate(pfuel):
                        input_state = {

                                        "P_fuel": z,
                                        "P_air": j,
                                        "T": i,
                                        "x": 0.999,
                                        "y": 0.21,
                                        
                                        "Vair": 305,
                                        "Vfuel": 84.5
                                        }
                        
                        nedstack = GenericMatlabModel(input_df = input_df, nominal_parameters = nominal_parameters)
                        test_df = nedstack.generate_input_signal(T = [input_state["T"]], P_fuel = [input_state["P_fuel"]], P_air = [input_state["P_air"]], 
                                                                 Vair = [input_state["Vair"]], Vfuel = [input_state["Vfuel"]], x = [input_state["x"]], 
                                                                 y = [input_state["y"]])
                        
                        nedstack.input_df = copy.deepcopy(test_df)

                        nedstack.dynamic_response(transfer_function="off")
                        nedstack_response = nedstack.response_df
                        
                       # if nedstack_response.iloc[1]["V_fc"]>55:
                            #and nedstack_response.iloc[1]["V_fc"]<70 and not isinstance(nedstack_response["V_fc"], complex):
                        V_fc_list = nedstack_response["V_fc"].tolist() 
                       
                        searched_value = {

                                        "P_fuel": z,
                                        "P_air": j,
                                        "T": i,
                                        # "x": 0.999,
                                        # "y": 0.21,
                                        
                                        # "Vair": 305,
                                        # "Vfuel": 84.5, 
                                        "V_fc": [V_fc_list],
                                        "I_fc": [input_df["I_fc"].tolist()]
                                        # "UfH2" : [UfH2_list],
                                        # "UfO2" : [UfO2_list]
                                        }
                        
                        df2 = pandas.DataFrame(searched_value, index=[0])
                        df = pandas.concat([df, df2], ignore_index=True)
            self.data_simulated = copy.deepcopy(df)
            
        elif fc_model=='StateSpaceModel_aug':
            P_fuel = [1.5, 3]
            P_air = [1, 3]
         
            temperature = [333, 373]
            pfuel = np.linspace(P_fuel[0], P_fuel[1], 10)
            pair = np.linspace(P_air[0], P_air[1], 10)
            # vair = np.linspace(Vair[0], Vair[1], 10)
            # vfuel = np.linspace(Vfuel[0], Vfuel[1], 10)
            T = np.linspace(temperature[0], temperature[1], 10)
            I_fc = np.linspace(0.01,200, 120)
            t = np.linspace(0, 500, 120)
            input_df = pandas.DataFrame()
            input_df["I_fc"] = I_fc
            input_df["t"] = t
            response_df = pandas.DataFrame()
            for indexi, i in enumerate(pfuel):
                for indexj , j in enumerate(pair):
                    
                    for indexz, z in enumerate(T):
                        input_state = {

                                        "P_fuel": i,
                                        "P_air": j,
                                        "T": z,
                                        "x": 0.999,
                                        "y": 0.21,
                                        
                                        "Vair": 305,
                                        "Vfuel": 84.5
                                        }
                        
                        nedstack = GenericMatlabModel(input_df = input_df)
                        test_df = nedstack.generate_input_signal(T = [input_state["T"]], P_fuel = [input_state["P_fuel"]], P_air = [input_state["P_air"]], 
                                                                 Vair = [input_state["Vair"]], Vfuel = [input_state["Vfuel"]], x = [input_state["x"]], 
                                                                 y = [input_state["y"]])
                        
                        quasi_input_df = copy.deepcopy(test_df)
                       
                        V_fc = []
                        for indexk, k in quasi_input_df.iterrows():
                            #P_H2, P_O2 = nedstack.calc_pressure(nom=False,util=False)
                            params["P_H2"] = input_state["P_fuel"]
                            params["P_O2"] = input_state["P_air"]
                            params["T"] = k["T"]
                            voltage = model_1.quasi_static_model(k["I_fc"], **params)
                            voltage= voltage
                            V_fc.append(voltage)
                            
                        response_dict = {"P_fuel":input_state["P_fuel"],
                                    "P_air":input_state["P_air"],
                                    "T": input_state["T"],
                                    "V_fc": [V_fc],
                                    "I_fc": [input_df["I_fc"].tolist()]}
                        
                        response = pandas.DataFrame(response_dict, index=[0])
                        response_df = pandas.concat([response_df, response], ignore_index=True)

            #response_df.to_csv("state_space_vis.csv", index=True)
            
            
    def clean_data(self):
        """
        

        Parameters
        ----------
        data : TYPE
            DESCRIPTION.

        Returns
        -------
        data : TYPE
            DESCRIPTION.

        """
        data = copy.deepcopy(self.data_simulated)
        data.dropna(inplace= True)
        
        return data
    
    # Works only for generic model and equivalent circuit for now
    def data_prep(self, window_size=5):
        """
        

        Parameters
        ----------
        interval : TYPE, optional
            DESCRIPTION. The default is 5.

        Returns
        -------
        None.

        """
        data_clean = self.clean_data()
        self.column_number = data_clean.columns.get_loc("V_fc")
        scaled_data = self.scale_data(data_clean)
        X_tensor, y_tensor = self.create_tensor(scaled_data, window_size)
        X_train, X_test, y_train, y_test = self.train_test_split(X_tensor, y_tensor)
        
        self.X_train_scaled = copy.deepcopy(X_train)
        self.X_test_scaled = copy.deepcopy(X_test)
        self.y_train_scaled = copy.deepcopy(y_train)
        self.y_test_scaled = copy.deepcopy(y_test)
        
    def data_prep_regression(self):
       
        data_clean = self.clean_data()
        data_clean.rename(columns={'V_fc': 'A', 'I_fc': 'C'}, inplace=True)
        data_clean = data_clean.explode(list('AC'))
        data_clean.rename(columns={'A': 'V_fc', 'C': 'I_fc'}, inplace=True)
        self.column_number = data_clean.columns.get_loc("V_fc")
        scaled_data = self.scale_data(data_clean)
        X = np.hstack((scaled_data[:, :self.column_number], scaled_data[:, self.column_number+1:]))
        y = scaled_data[:, self.column_number]
        X_train, X_test, y_train, y_test = self.train_test_split(X, y)
        self.scaled_data = copy.deepcopy(scaled_data)
        self.X_train_scaled = copy.deepcopy(X_train)
        self.X_test_scaled = copy.deepcopy(X_test)
        self.y_train_scaled = copy.deepcopy(y_train)
        self.y_test_scaled = copy.deepcopy(y_test)
        
        
    
    def scale_data(self, data):
        """
        

        Parameters
        ----------
        data : TYPE
            DESCRIPTION.

        Returns
        -------
        scaled_data : TYPE
            DESCRIPTION.

        """
        
        scaler = MinMaxScaler()
        scaler = scaler.fit(data)
        scaled_data = scaler.transform(data)
        self.scaler = copy.deepcopy(scaler)
        
        return scaled_data
    
    def create_tensor(self, data, window_size):
        """
        

        Parameters
        ----------
        data : TYPE
            DESCRIPTION.
        window_size : TYPE
            DESCRIPTION.

        Returns
        -------
        X : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.

        """
        X = []
        y = []
        
        
        predict_len = 1
        
        for i in range(window_size, len(data)-predict_len+1):
            X.append(data[i-window_size:i, 0:data.shape[1]])
            y.append(data[i+predict_len-1:i+predict_len, self.column_number])
        
        X, y = np.array(X), np.array(y)
        
        return X, y
    
    def train_test_split(self, X, y):
        """
        

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.

        Returns
        -------
        X_train : TYPE
            DESCRIPTION.
        X_test : TYPE
            DESCRIPTION.
        y_train : TYPE
            DESCRIPTION.
        y_test : TYPE
            DESCRIPTION.

        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle = False)
   
        return X_train, X_test, y_train, y_test
    
    def inverse_transform(self, predictions, scaler):
        """
        

        Returns
        -------
        predictions_unscaled : TYPE
            DESCRIPTION.

        """
        if isinstance(predictions, type(None)):
            predictions = tensor_data.predicted_data
        predictions_copy = np.repeat(predictions, self.data_simulated.shape[1], axis = -1 )
        predictions_unscaled = self.scaler.inverse_transform(predictions_copy)[:, self.column_number]
    
        return predictions_unscaled
    
    def inverse_transform_regression(self, predictions, data=None):
        
        if isinstance(data, type(None)):
            predictions_copy = np.repeat(predictions, self.data_simulated.shape[1], axis = -1 )
            predictions_unscaled = self.scaler.inverse_transform(predictions_copy)[:, self.column_number]
            
        else:
            predictions_unscaled = self.scaler.inverse_transform(data)
        
        return predictions_unscaled
    
    def initialise_tensor_data(self):
        """
        

        Returns
        -------
        None.

        """
        tensor_data.X_train = copy.deepcopy(self.X_train_scaled)
        tensor_data.X_test = copy.deepcopy(self.X_test_scaled)
        tensor_data.scaler = copy.deepcopy(self.scaler)
        tensor_data.y_train = copy.deepcopy(self.y_train_scaled)
        tensor_data.y_test = copy.deepcopy(self.y_test_scaled)
    
    def build_data(self, X_train, predictions):
        data= np.insert(X_train, self.column_number, predictions, axis=1)
        return data
    
#%% Neural network models    
class LSTM_model:
    
    #incorporate tuner and learning_rate
    def __init__(self, learning_rate=None, noise=False):
        self.model = None
        self.learning_rate = learning_rate
        self.noise = noise
        self.history = None
        self.predictions = None
        
    def build_architecture(self):
        """
        

        Returns
        -------
        model : TYPE
            DESCRIPTION.

        """
        model = Sequential()
        model.add(LSTM(64, activation='relu', 
                       input_shape=(tensor_data.X_train.shape[1],
                                    tensor_data.X_train.shape[2]), 
                                       return_sequences=True))
        model.add(LSTM(32, activation='relu', return_sequences=False))
        model.add(Dropout(0.2))
        if self.noise == True:
            model.add((GaussianNoise(0.1)))
            
        model.add(Dense(tensor_data.y_train.shape[1]))
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        self.model = copy.deepcopy(model)
        return model
    
    def train_model(self, batch_size=16, model_name="LSTM-untuned", X_train=None, y_train=None):
        """
        

        Returns
        -------
        None.

        """
        model = self.build_architecture()
        # stop_early = tf.keras.callbacks.EarlyStopping(monitor= 'val_loss', 
        #                                               patience=3)
        
        self.history = model.fit(X_train, y_train, epochs=500,
                            batch_size=batch_size, validation_split=0.2, verbose=1,
                            )
        
        #callbacks = [stop_early]
        model.save('{.h5}'.format(model_name))
        
    def model_predict(self, X_train, model_name="LSTM-untuned"):
        """
        

        Returns
        -------
        input_df : TYPE
            DESCRIPTION.
        output_df : TYPE
            DESCRIPTION.
        predictions_df : TYPE
            DESCRIPTION.

        """
        self.train_model()
        model = tf.keras.models.load_model('{}.h5'.format(model_name))
        
        predictions = model.predict(X_train)
        
        self.predictions = copy.deepcopy((predictions))
        
        tensor_data.predicted_data = copy.deepcopy((predictions))
        
        return predictions
    
    def visualize_loss(self):
        """
        

        Returns
        -------
        None.

        """
        plt.plot(self.history.history['loss'], label='train')
        plt.plot(self.history.history['val_loss'], label='test')
        plt.legend()
        plt.show()
        
class ANN_model():
    def __init__(self, learning_rate=None, noise=False):
        
        self.is_scaled=None
        self.model = None
        self.learning_rate = learning_rate
        self.noise = noise
        self.history = None
        self.predictions = None
        
    def build_architecture(self,input_size=4):
        model = Sequential([
                            Dense(32, activation='relu', input_shape=(input_size,)),
                            Dense(64, activation='relu'),
                            Dense(1, activation='sigmoid')  # For binary classification, change to 'softmax' for multi-class
                            ])
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        self.model=copy.deepcopy(model)
        return model
        
    def train_model(self, batch_size=16, model_name="LSTM-untuned", X_train=None, y_train=None):
        """
        

        Returns
        -------
        None.

        """
        model = self.build_architecture()
        # stop_early = tf.keras.callbacks.EarlyStopping(monitor= 'val_loss', 
        #                                               patience=3)
        
        self.history = model.fit(X_train, y_train, epochs=10,
                            batch_size=batch_size, validation_split=0.2, verbose=1,
                            )
        
        #callbacks = [stop_early]
        model.save('{}.h5'.format(model_name))
        
    def model_predict(self, X_train, model_name="LSTM-untuned"):
        """
        

        Returns
        -------
        input_df : TYPE
            DESCRIPTION.
        output_df : TYPE
            DESCRIPTION.
        predictions_df : TYPE
            DESCRIPTION.

        """
        
        model = tf.keras.models.load_model('{}.h5'.format(model_name))
        
        predictions = model.predict(X_train)
        
        self.predictions = copy.deepcopy((predictions))
        
        tensor_data.predicted_data = copy.deepcopy((predictions))
        
        return predictions
    
    def visualize_loss(self):
        """
        

        Returns
        -------
        None.

        """
        plt.plot(self.history.history['loss'], label='train')
        plt.plot(self.history.history['val_loss'], label='test')
        plt.legend()
        plt.show()
        
        