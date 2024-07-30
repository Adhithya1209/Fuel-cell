# -*- coding: utf-8 -*-
"""
Test Script for Fuel cell related stuff

"""

import sys
import numpy as np
import pandas as pd
import copy

import scipy.stats as st
from scipy.optimize import curve_fit

from matplotlib import gridspec
from matplotlib import pyplot as plt

import aging.fuel_cell_param_id as fuel_cell_param_id


#%% Basic Test Run
polar_curve = pd.read_csv('aging/nedstack_ps6_datasheet.csv', names=['current','voltage'], header=None)
polar_curve.drop(0, inplace=True)
polar_curve.reset_index(drop=True,inplace=True)
polar_curve["current"] = polar_curve["current"].astype(float)
polar_curve["voltage"] = polar_curve["voltage"].astype(float)
current = polar_curve['current'].to_numpy()
model_1 = fuel_cell_param_id.E_Act_Ohm_Conc_Model()

model_1.load_polar_curve(polar_curve)
model_1.fit_params(use_polar_curve=True)

params = model_1.params

fig = model_1.plot_compare_polar_curve()


#%% Ageing param visualizer

key = 'n_ohm_1'
scale_fac = [1, 2, 3]

params_1 = copy.deepcopy(params)
params_1[key] = scale_fac[0] * params_1[key]

params_2 = copy.deepcopy(params)
params_2[key] = scale_fac[1] * params_2[key]

params_3 = copy.deepcopy(params)
params_3[key] = scale_fac[2] * params_3[key]

params_set = [params_1, params_2, params_3]

fig, ax = model_1.plot_model_params(current, params_set, key)


#%% 2016 ageing paper 0, 1000, 2000 h test

polar_curve_ageing = pd.read_csv('2016 paper polar ageing.csv')


polar_0h = polar_curve_ageing.iloc[1:,[0,1]]
polar_0h.columns = ['current','voltage']
polar_0h.dropna(inplace=True)
polar_0h.reset_index(drop=True, inplace=True)
polar_0h = polar_0h.astype({'current':'float','voltage':'float'})

model_0h = fuel_cell_param_id.E_Act_Ohm_Conc_Model()

model_0h.load_polar_curve(polar_0h)
model_0h.fit_params(use_polar_curve=True)

params_0h = model_0h.params

fig_0h = model_0h.plot_compare_polar_curve()



polar_1000h = polar_curve_ageing.iloc[1:,[2,3]]
polar_1000h.columns = ['current','voltage']
polar_1000h.dropna(inplace=True)
polar_1000h.reset_index(drop=True, inplace=True)
polar_1000h = polar_1000h.astype({'current':'float','voltage':'float'})

model_1000h = fuel_cell_param_id.E_Act_Ohm_Conc_Model()

model_1000h.load_polar_curve(polar_1000h)
model_1000h.fit_params(use_polar_curve=True)

params_1000h = model_1000h.params

fig_1000h = model_1000h.plot_compare_polar_curve()



polar_2000h = polar_curve_ageing.iloc[1:,[4,5]]
polar_2000h.columns = ['current','voltage']
polar_2000h.dropna(inplace=True)
polar_2000h.reset_index(drop=True, inplace=True)
polar_2000h = polar_2000h.astype({'current':'float','voltage':'float'})

model_2000h = fuel_cell_param_id.E_Act_Ohm_Conc_Model()

model_2000h.load_polar_curve(polar_2000h)
model_2000h.fit_params(use_polar_curve=True)

params_2000h = model_2000h.params

fig_2000h = model_2000h.plot_compare_polar_curve()

#%% 2016 ageing paper 0, 1000, 2000 h test - ageing analyser

polar_curve_database = {
    0 : polar_0h,
    1000 : polar_1000h,
    2000 : polar_2000h
    }

