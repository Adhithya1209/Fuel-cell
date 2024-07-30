# -*- coding: utf-8 -*-
"""
Fuel Cell Param Identifier
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import sys
import pickle
import copy
from dataclasses import dataclass, field


import numpy as np
import pandas as pd

import scipy.stats as st
from scipy.optimize import curve_fit

from matplotlib import gridspec
from matplotlib import pyplot as plt

#%% Utility Functions

def ageing_analyser(polar_curve_dict, model_name):
    
    pass    




#%% Base Class Definitions

@dataclass
class FuelCellData:
    """
    Data Class for transfering data between different models and classes
    
    Important Attributes:
        
        - data [ pd.DataFrame ] : Unaltered copy of data file.
        
        - data_scaled [ pd.DataFrame ] : Copy of data file on which transformations are made.
        
        - feature_names [ list(string) ] : Signal names that should be inputs to the models.
        
        - target_names [ list(string) ] : Signal names that should be outputs of the models.
        
        - ranges [ dict ] : Dictionary containing permissible ranges of signals to scaled down and up.
        
        - is_scaled [ dict ] : Dictionary of booleans indincating whether a signal is scaled or not.
        
        - tensors [ dict ] : Dicitonary of important attributes to be used in the model.
    
    """
    
    ### Fuel Cell Data ###
    data : pd.DataFrame = field(default_factory=pd.DataFrame)
    data_scaled : pd.DataFrame = field(default_factory=pd.DataFrame)
    
    feature_names : list = field(default_factory=list)
    target_names : list = field(default_factory=list)
    
    ranges : dict = field(default_factory=dict)
    is_scaled : dict = field(default_factory=dict)
    
    tensors : dict = field(default_factory=dict)



class FuelCellStatic:
    """
    Contains base definition for all Static Fuel Cell Models - Typically trained with Polarization Curves
    
    Important Attributes:
        
        - polar_curve [ pd.DataFrame ] : Dataframe containing Polarizaiton Curve
        - params [ dict ] : Contains fuel cell parameters
        - param_bounds [ dict ] : Contains fuel cell paramters bounds
        
            
    """
    
    def __init__(self):
        
        
        self.polar_curve = pd.DataFrame()
        
        self.params = {}
        self.param_bounds = {}
        
        
    def fit_params(self, xdata=None, ydata=None, use_polar_curve=True):
        
        param_bounds = copy.deepcopy(self.param_bounds)
        polar_curve = copy.deepcopy(self.polar_curve)
        
        if use_polar_curve:
            xdata, ydata = polar_curve['current'].to_numpy(), polar_curve['voltage'].to_numpy()
        
        popt, pcov = curve_fit(self.quasi_static_model, xdata, ydata,
                               bounds=(tuple([v[0] for k, v in param_bounds.items()]),
                                       tuple([v[1] for k, v in param_bounds.items()])),
                               method='trf')
        
        self.params = dict(zip(self.param_bounds.keys(), popt))
        
        return popt, pcov
    
    
    def plot_compare_polar_curve(self):
        
        polar_curve = copy.deepcopy(self.polar_curve)
        params = copy.deepcopy(self.params)
        
        current, voltage = polar_curve['current'].to_numpy(), polar_curve['voltage'].to_numpy()
        voltage_pred = self.quasi_static_model(current, **params)
        
        
        fig = plt.figure()
        fig.suptitle(" Plot Comparison ")
        
        ax = fig.add_subplot(111)
        ax.plot(current, voltage, 'rx', label='polarisation curve (ANN predicted)')
        ax.plot(current, voltage_pred, 'b--', label='polarisation curve (synthetic data)')
        ax.set_xlabel("current")
        ax.set_ylabel("voltage")
        ax.legend()
        
        
        return fig


class FuelCellDynamic(FuelCellStatic):
    
    def __init__(self):
        
        self.data = pd.DataFrame()
        
    
    

#%%% Models

class E_Act_Ohm_Conc_Model(FuelCellStatic):
    """
    Model Equations - Low Temeprature PEMFC Stack Model
    
    E_ocv = V_ocv_1  +  V_ocv_2 * T  + V_ocv_3 * T * ln(P_H2 * P_O2^0.5)
    n_act = n_act_1 * T * ln(i / i_0)
    n_ohm = n_ohm_1 * i
    n_conc = n_conc_1 * T * ln(1 - (i / i_L)) 
    
    E_FC = E_ocv - n_act - n_ohm - n_conc
    
    """
    
    def __init__(self, **kwargs):
        
        super(E_Act_Ohm_Conc_Model, self).__init__(**kwargs)
        
        
        
    def quasi_static_model(self, i, T, P_H2, P_O2, V_ocv_1, V_ocv_2, V_ocv_3, n_act_1, i_0, n_ohm_1, n_conc_1, i_L):

        E_ocv = V_ocv_1 + (V_ocv_2 * T) + (V_ocv_3 * T * np.log(P_H2 * (P_O2**0.5)))
        n_act = n_act_1 * T * np.log((i + 1e-5) / i_0)
        n_ohm = n_ohm_1 * i
        n_conc = n_conc_1 * T * np.log(1 - (i / i_L))
        
        E_FC = E_ocv - n_act - n_ohm - n_conc
        
        return E_FC/0.5
    

    def load_polar_curve(self, polar_curve:pd.DataFrame, T=333, P_H2=0.99, P_O2=0.21, tol=1e-5):
        
        min_current, min_voltage = polar_curve.min(axis=0)
        max_current, max_voltage = polar_curve.max(axis=0)
        
        param_bounds = {
            'T' : [T-tol, T+tol],
            'P_H2' : [P_H2-tol, P_H2+tol],
            'P_O2' : [P_O2-tol, P_O2+tol],
            'V_ocv_1' : [max_voltage-tol, max_voltage+tol],
            'V_ocv_2' : [-np.inf, 0],
            'V_ocv_3' : [0, np.inf],
            'n_act_1' : [-np.inf, np.inf],
            'i_0' : [tol, 0.5*max_current],
            'n_ohm_1' : [0, np.inf],
            'n_conc_1' : [-np.inf, 0],
            'i_L' : [max_current, max_current*1.5]
            }

        self.param_bounds = param_bounds
        self.polar_curve = polar_curve



    def score(self, polar_curve, params):
        pass
    
    
    
    def plot_model_params(self, current, params_set, key):
        
        fig = plt.figure()
        fig.suptitle(" Plot Comparison ")
        
        ax = fig.add_subplot(111)
         
        for params in params_set:
            
            voltage_pred = self.quasi_static_model(current, **params)
            params[key] = round(params[key], 3)
            
            ax.plot(current, voltage_pred, label=key+" : "+str(params[key]))
            ax.legend()
            
        return fig, ax
            



class SS_E_Act_Ohm_Conc_Model(FuelCellDynamic):
    
    pass
        
            