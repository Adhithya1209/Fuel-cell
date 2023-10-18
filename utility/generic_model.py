# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 14:26:42 2023

@author: jayaraman
"""
"""
Parameters Nedstack PS6
"""
import pandas

input_df = pandas.read_csv("input_current_time.csv")

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
                      "Td" : 1,
                      "ufo2_peak" : 67,
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
                }