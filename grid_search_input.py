# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 16:47:28 2023

@author: jayaraman
"""
import numpy as np
import pandas
import fuel_cell
import copy
import warnings
warnings.filterwarnings("ignore")

#%% Initialise Inputs nedstack ps6
P_fuel = [0.75, 1.75]
P_air = [0.75, 1.5]
Vair = [270, 330]
Vfuel = [75, 270]
temperature = [290, 350]
pfuel = np.linspace(P_fuel[0], P_fuel[1], 5)
pair = np.linspace(P_air[0], P_air[1], 5)
vair = np.linspace(Vair[0], Vair[1], 5)
vfuel = np.linspace(Vfuel[0], Vfuel[1], 5)
T = np.linspace(temperature[0], temperature[1], 5)
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

#%% Search optimal value nedstack ps6
for indexi, i in enumerate(pfuel):
    for indexj , j in enumerate(pair):
        for indexx , x in enumerate(vair):
            for indexy, y in enumerate(vfuel):
                for indexz, z in enumerate(T):
                    input_state = {
    
                                    "P_fuel": i,
                                    "P_air": j,
                                    "T": z,
                                    "x": 0.999,
                                    "y": 0.21,
                                    
                                    "Vair": x,
                                    "Vfuel": y
                                    }
                    
                    nedstack = fuel_cell.GenericMatlabModel(input_df = input_df, nominal_parameters = nominal_parameters)
                    test_df = nedstack.generate_input_signal(T = [input_state["T"]], P_fuel = [input_state["P_fuel"]], P_air = [input_state["P_air"]], 
                                                             Vair = [input_state["Vair"]], Vfuel = [input_state["Vfuel"]], x = [input_state["x"]], 
                                                             y = [input_state["y"]])
                    
                    nedstack.input_df = copy.deepcopy(test_df)
    
                    nedstack.dynamic_response(x0 = -32.5, transfer_function="on")
                    nedstack_fc = nedstack.fuel_cell_parameters
                    nedstack_response = nedstack.response_df
                    nedstack_calc = nedstack.calculated_space
                    
                    if nedstack_response.iloc[1]["V_fc"]>55 and nedstack_response.iloc[1]["V_fc"]<75 and not isinstance(nedstack_response["V_fc"], complex):
                        V_fc_list = nedstack_response["V_fc"].tolist() 
                        searched_value = {
    
                                        "P_fuel": i,
                                        "P_air": j,
                                        "T": z,
                                        "x": 0.999,
                                        "y": 0.21,
                                        
                                        "Vair": x,
                                        "Vfuel": y, 
                                        "V_fc": [V_fc_list]
                                        }
                        
                        df2 = pandas.DataFrame(searched_value, index=[0])
                        df = pandas.concat([df, df2], ignore_index=True)

#%% Initialise input parameters - Nedstack ps6 steady state model
PH2 = [0.75, 2.5]
PO2 = [0.75, 2.5]

temperature = [290, 600]
ph2 = np.linspace(PH2[0], PH2[1], 10)
po2 = np.linspace(PO2[0], PO2[1], 10)

T = np.linspace(temperature[0], temperature[1], 100)
I_fc = np.linspace(0.01,200, 120)

input_df = pandas.DataFrame()
input_df["I_fc"] = I_fc

#%% Search Nedstack Ps6 Steady state model

df = pandas.DataFrame()

for indexi, i in enumerate(ph2):
    for indexj, j in enumerate(T):
        for indexx, x in enumerate(po2):
            nedstack = fuel_cell.SteadyStateEmprical()
            nedstack.generate_input_signal(I=[5,200], T=[j], PH2=[i], PO2=[x])

            response_df = nedstack.run_steady_state()
            if response_df.iloc[1]["Vfc"]>55 and response_df.iloc[1]["Vfc"]<60 and not isinstance(response_df["Vfc"], complex):
                V_fc_list = response_df["Vfc"].tolist() 
                searched_value = {
    
                                "PH2": i,
                                "PO2": x,
                                "T": j,
                                "V_fc": [V_fc_list]
                                }
                
                df2 = pandas.DataFrame(searched_value, index=[0])
                df = pandas.concat([df, df2], ignore_index=True)
