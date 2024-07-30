# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 15:36:16 2023

@author: jayaraman

State space model testing
"""
import numpy as np

import pandas
import copy

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
    

import warnings

from scipy.integrate import odeint


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
    
    
    def model(self, x, t):
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
        self.counter = self.counter+1
        self.input_states["I"] = I
        
        self.input_states["u_Tr"] = u_Tr
        self.input_states["u_Pa"] = u_Pa
        self.input_states["u_Pc"] = u_Pc
        
        return [u_Pa, u_Pc, u_Tr]
    
    
    #### Solver and Simulation
    def ode_solver(self, initial_state_vector = None, solver = "odeint"):
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
        t = self.input_df["t"]
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
        
class statespacetestmodel:
    def __init__(self):
        self.a_matrix = None
        self.solver = None
        
    def calc_matrix_a(self):
        
    
    