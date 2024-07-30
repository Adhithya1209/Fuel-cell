# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 17:18:59 2023

@author: jayaraman
"""
import numpy as np
from scipy.integrate import odeint

import pandas
import copy
import warnings
from scipy.integrate import solve_ivp


def generate_step(t_span, t_step, num, signal_value, signal_amp=5):
    t_signal = np.linspace(t_span[0], t_span[1], num)
    signal = pandas.DataFrame()
    i_signal = []
    for t in t_signal:
        if t> t_step:
            i = signal_value
            i_signal = np.append(i_signal, i)
            
        else:
            i = signal_amp
            i_signal = np.append(i_signal, i)
            
    signal["t"] = t_signal
    signal["I_fc"] = i_signal
    
    return signal
            
        
def dum_input_current(t=None):
    if isinstance(t, type(None)):
        t_span = (90,110)
        t_step = 100
        signal = generate_step(t_span, t_step, 100, 25)
        
    return signal
    




#%% fuel cell simple model 1 - solve_ivp
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
        self.cell_parameters = {"A_s": 3.2 * (10**(-2)),
                                "a": -3.08 * (10**(-3)),
                                "a0": 1.3697,
                                "b": 9.724 * (10**(-5)),
                                "Cfc":500,
                                "C": 10,
                                "E0_cell": 1.23,
                                "e": 2,
                                "F": 96487,
                                "del_G": 237.2 * (10**(3)),
                                "hs": 37.5,
                                "K_I": 1.87*(10**(-3)),
                                "K_T": -2.37 * (10**(-3)),
                                "M_fc": 44,
                                "mH2O": 8.614 * (10**(-5)),
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

                                "T": None,
                                "PH2": None,
                                "PO2": None,
                                }
        self.current_theta = None
        
        self.input_states = {   
                                "u_Pa": None,
                                "u_Pc": None,
                                "u_Tr": None
                               }
        
        self.current_input = None
        self.matrix_a_comp = None
        self.matrix_b_comp = None
        self.matrix_g_comp = None
        
        if isinstance(input_df, type(None)):
            self.input_df = pandas.DataFrame()
            
        else:
            self.input_df = copy.deepcopy(input_df)
            
            
        self.output_df = pandas.DataFrame()
        
        if isinstance(current_profile, type(None)):
            self.input_signal = dum_input_current()
            
        self.x = None
        self.calc_space={"activation": [],
                         "ohmic":[],
                         "dT_dt": [],
                         "theta_8":[]}

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
    
    
    #### Polarisation loss 
    def activation_loss(self, I=None, T=None):
        """
        

        Returns
        -------
        Vact : TYPE
            DESCRIPTION.

        """
        
        cell_parameters = copy.deepcopy(self.cell_parameters)
        if isinstance(I, type(None)):
            I = self.current_input
        if isinstance(T, type(None)):
            T = self.current_state_variables["T"]
        a = cell_parameters["a"]
        b = cell_parameters["b"]
        a0 = cell_parameters["a0"]
        
        
        Vact = a0 + T * (a + (b*np.log(I)))
        self.calc_space["activation"].append(Vact)
        return Vact
    
    
    def ohmic_loss(self,I=None, T=None):
        """
        

        Returns
        -------
        Ro : TYPE
            DESCRIPTION.

        """
        if isinstance(I, type(None)):
            I = self.current_input
        if isinstance(T, type(None)):
            T = self.current_state_variables["T"]
        cell_parameters = copy.deepcopy(self.cell_parameters)
        Roc = cell_parameters["Roc"]
        K_I = cell_parameters["K_I"]
        K_T = cell_parameters["K_T"]
        Ro = Roc + K_I*I + K_T * T
        Vo = Ro * I
        self.calc_space["ohmic"].append(Vo)
 
        return Vo
    
    #### Matrix calculation
    def calc_theta(self):
        """
        

        Returns
        -------
        None.

        """
        I = self.current_input
        current_state_variables = copy.deepcopy(self.current_state_variables)
        cell_parameters = copy.deepcopy(self.cell_parameters)
        R = cell_parameters["R"]
        Va = cell_parameters["Va"]
        mH2Oa_in = cell_parameters["mH2O"]
        PH2Oa_in = cell_parameters["PH2O_e"]
        F = cell_parameters["F"]
        ns = cell_parameters["ns"]

        x3 = current_state_variables["T"]
        x4 = current_state_variables["PH2"]
        x5 = current_state_variables["PO2"]

        
        theta_1 = (R*mH2Oa_in*x3)/(Va*PH2Oa_in)
        theta_2 = (R*x3)/(2*Va*F)
        
        Vc = cell_parameters["Vc"]
        mH2Oc_in = cell_parameters["mH2O"]
        PH2Oc_in = cell_parameters["PH2O_e"]
        theta_3 = (R*mH2Oc_in*x3)/(Vc*PH2Oc_in)
        theta_4 = (R*x3)/(4*Vc*F)
        Vo = self.ohmic_loss()

        Vact = self.activation_loss()
        E0_cell = cell_parameters["E0_cell"]

        Mfc = cell_parameters["M_fc"]
        Cfc = cell_parameters["Cfc"]
        theta_7 = ns*(E0_cell + (R*x3)/(2*F) * np.log(x4*(x5**0.5)) - Vact - Vo)
        theta_8 = ns*((2*E0_cell/Mfc*Cfc) + (R*x3/F*Mfc*Cfc)*np.log(x4*(x5**0.5)) - Vact  - Vo)
        self.current_theta = {"theta_1": theta_1,
                      "theta_2": theta_2,
                      "theta_3": theta_3,
                      "theta_4": theta_4,
                      "theta_7": theta_7,
                      "theta_8": theta_8}   
        self.calc_space["theta_8"].append(theta_8)
        
    def matrix_a(self):
        """
        

        Returns
        -------
        a_matrix : TYPE
            DESCRIPTION.

        """
    
        cell_parameters = copy.deepcopy(self.cell_parameters)
     
        a_matrix = np.zeros((5,5))
        a_44 = self.matrix_a_44
        
        cell_parameters = copy.deepcopy(self.cell_parameters)
        self.calc_theta()
        theta = copy.deepcopy(self.current_theta)
        
        theta_1 = theta["theta_1"]
        
        theta_3 = theta["theta_3"]
        lamb_a = cell_parameters["lamb_a"]
        lamb_c = cell_parameters["lamb_c"]
      
        for index_i, row in enumerate(a_matrix):
            for index_j, element in enumerate(row):
                if index_i == index_j:
                    if index_j == 0:
                        a_matrix[index_i][index_j] = (-1/lamb_c)
                        
                    elif index_i == 1:
                        a_matrix[index_i][index_j] = (-1/lamb_a)
                        
                    elif index_i == 2:
                        a_matrix[index_i][index_j] = a_44
                        
                    elif index_i == 3:
                        a_matrix[index_i][index_j] = (-2) * theta_1
                        
                    elif index_i == 4:
                        a_matrix[index_i][index_j] = (-2) * theta_3
        
        self.matrix_a_comp = a_matrix
        return a_matrix
        
        
    def matrix_b(self):
        """
        

        Returns
        -------
        b_matrix : TYPE
            DESCRIPTION.

        """
        
        b_matrix = np.zeros((5,3))
        
        b_22 = -1 * self.matrix_a_44
        self.calc_theta()
        theta = copy.deepcopy(self.current_theta)
        theta_1 = theta["theta_1"]
        theta_3 = theta["theta_3"]

        
        b_matrix[2][2] = b_22
        b_matrix[3][0] = 2*theta_1
        b_matrix[4][1] = 2*theta_3
        
        self.matrix_b_comp = b_matrix
        return b_matrix
        
    def matrix_g(self):
        """
        

        Returns
        -------
        g_matrix : TYPE
            DESCRIPTION.

        """
        cell_parameters = copy.deepcopy(self.cell_parameters)
        g_matrix = np.zeros(5)
        lamb_c = cell_parameters["lamb_c"]
        lamb_a = cell_parameters["lamb_a"]
        F = cell_parameters["F"]

        
        current_theta = copy.deepcopy(self.current_theta)
        theta_2 = current_theta["theta_2"]
        theta_4 = current_theta["theta_4"]

        theta_8 = current_theta["theta_8"]
        
        g_matrix[0] = 1/(4*lamb_c*F)
        g_matrix[1] = 1/(2*lamb_a*F)
        g_matrix[2] = -theta_8
        g_matrix[3] = -theta_2
        g_matrix[4] = -theta_4

        
        g_matrix_T = np.transpose(g_matrix)
        self.matrix_g_comp = g_matrix_T
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
                                "T": x[2],
                                "PH2": x[3],
                                "PO2": x[4],
                                }
        a_matrix = self.matrix_a()
        
        b_matrix = self.matrix_b()
        g_matrix_T = self.matrix_g()
        
        dx_dt = a_matrix@x + (b_matrix @ u).flatten() + self.current_input*(g_matrix_T)
        self.calc_space["dT_dt"].append(dx_dt[2])
        
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
        closest_row_index = (input_df["t"] - t).abs().idxmin()
        u_Pa = input_df.at[closest_row_index, "u_Pa"]
        u_Pc = input_df.at[closest_row_index, "u_Pc"]
        u_Tr = input_df.at[closest_row_index, "u_Tr"]
        I = input_df.at[closest_row_index, "I_fc"]

        self.current_input = I
        
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
                    "T": 308,
                    "PH2": 2,
                    "PO2": 1,
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
                u_Pa = l*[2]
                
            try:
                if len(kwargs["u_Pc"]) ==1:
                    u_Pc = l*[kwargs["u_Pc"]]
                    
                else:
                    raise ValueError("u_Pc signal cannot be varied in Default"\
                                     "mode. Choose profile as linear or step to modify value")
                
            except KeyError:
                u_Pc = l*[1]
                
            
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
                                "T": None,
                                "PH2": None,
                                "PO2": None,
                                }
        self.current_theta = None