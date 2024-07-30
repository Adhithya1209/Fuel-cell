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
import matplotlib.pyplot as plt
#%% Initialise Inputs nedstack ps6
P_fuel = [1.5, 3]
P_air = [1, 3]
Vair = [250, 500]
Vfuel = [1, 84.5]
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

#%% Search optimal value nedstack ps6
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
            
            nedstack = fuel_cell.GenericMatlabModel(input_df = input_df, nominal_parameters = nominal_parameters)
            test_df = nedstack.generate_input_signal(T = [input_state["T"]], P_fuel = [input_state["P_fuel"]], P_air = [input_state["P_air"]], 
                                                     Vair = [input_state["Vair"]], Vfuel = [input_state["Vfuel"]], x = [input_state["x"]], 
                                                     y = [input_state["y"]])
            
            nedstack.input_df = copy.deepcopy(test_df)

            nedstack.dynamic_response(transfer_function="off")
            nedstack_fc = nedstack.fuel_cell_parameters
            nedstack_response = nedstack.response_df
            nedstack_calc = nedstack.calculated_space
            
           # if nedstack_response.iloc[1]["V_fc"]>55:
                #and nedstack_response.iloc[1]["V_fc"]<70 and not isinstance(nedstack_response["V_fc"], complex):
            V_fc_list = nedstack_response["V_fc"].tolist() 
            UfH2_list = nedstack_response["UfH2"].tolist()
            UfO2_list = nedstack_response["UfO2"].tolist()
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

#df.to_csv("GMM_nedstack-vis.csv", index=True)
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
#%% visualize GMM

# gmm_response = pd.read_csv("GMM_nedstack-vis.csv", index_col=None)
# gmm_response.drop(gmm_response.columns[0], axis=1, inplace=True)
# gmm_response["V_fc"] = gmm_response["V_fc"].astype(float)
# gmm_response["I_fc"] = gmm_response["I_fc"].astype(float)
# gmm_response["T"] = gmm_response["V_fc"].astype(float)
gmm_response = copy.deepcopy(df)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for index, row in gmm_response.iterrows():
    V_fc_list = row["V_fc"]
    I_fc_list = row["I_fc"]
    T_list = len(V_fc_list)*[row["P_fuel"]]
    
    ax.plot(I_fc_list,V_fc_list , T_list)
    
ax.set_xlabel('current')
ax.set_ylabel('voltage')
ax.set_zlabel('P_fuel')
plt.show()
#%% polarisation2d GMM
gmm_response = copy.deepcopy(df)

for index, row in gmm_response.head(5).iterrows():
    V_fc_list = row["V_fc"]
    I_fc_list = row["I_fc"]
    T= row["P_fuel"]

    plt.plot(I_fc_list,V_fc_list , label='{}bar'.format(T))
    
plt.xlabel('current')
plt.ylabel('voltage')
plt.title('Polarisation curve')
plt.legend()
plt.show()
