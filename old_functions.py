# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 10:38:24 2023

@author: jayaraman
"""


import numpy as np
import random
import pandas
import copy
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time
from dataclasses import dataclass, field
import opem
from opem.Dynamic.Padulles2 import Dynamic_Analysis
from sklearn.metrics import accuracy_score
import keras_tuner as kt
import os
from scipy import signal
from datetime import datetime, timedelta
import warnings

class PadullesDynamic1(CorreaModel):
    
    def __init__(self, current_lower, current_upper):
        super().__init__(current_lower, current_upper)
        self.test = None
        self.data = pandas.DataFrame()
        self.scaler = None
        self.model = None
        self.input_df = pandas.DataFrame()
        self.output_df = pandas.DataFrame()
        self.input_scaled = pandas.DataFrame()
        self.predictions_df = pandas.DataFrame()
        
    def test_vector(self):
        self.test = {
            
            "B": 0.04777,
            "C": 0.0136,
            "E0": 0.6,
            "KH2": 0.0000422,
            "KH2O": 0.000007716,
            "KO2": 0.0000211,
            "N0": 5,
            
            "Rint": 0.00303,
            "T": 343,
            "i-start": 0.1,
            
            "i-step": 0.1,
            "i-stop": 100,
            "qH2": 0.0004,
            "rho": 1.168,
            
            "tH2": 3.37,
            "tH2O": 18.418,
            "tO2": 6.74,
            
            "Name": "Padulles2_Test"}
        
    def run_padulles(self, vector = None):
        self.test_vector()
        
        if not vector:
            vector = self.test

            data=Dynamic_Analysis(InputMethod= vector,TestMode=True,
                                  PrintMode=True,ReportMode=True)
            
        else:
            data=Dynamic_Analysis(InputMethod= vector,TestMode=False,
                                  PrintMode=True,ReportMode=True)
            
        return data
            
        
    def data_prep(self):
        self.data = pandas.read_csv('dynamic_padulles2.csv')
        
    
    def scaling_data(self):
        
        self.data_prep()
        df = copy.deepcopy(self.data)
        input_df = df[['I (A)', 'PH2 (atm)', 'PH2O (atm)', 'PO2 (atm)'] ]
        output_df = df[['FC Voltage (V)', 'FC Efficiency ()', 'FC Power (W)']]
        
        self.input_df = copy.deepcopy((input_df))
        self.output_df = copy.deepcopy(output_df)
        self.scaler = StandardScaler()
        
        input_scaled = pandas.DataFrame(self.scaler.fit_transform(input_df), 
                                        columns = input_df.columns)
        
        output_scaled =  pandas.DataFrame(self.scaler.fit_transform(output_df), 
                                          columns = output_df.columns)
        
        self.input_scaled = copy.deepcopy(input_scaled)
        
        return input_scaled, output_scaled
        
    def test_train_split(self):
        
        input_scaled, output_scaled = self.scaling_data()
        
        X_train, X_test, y_train, y_test = train_test_split(
            input_scaled, output_scaled, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test
    
    def neural_network_core(self):
        
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(units=64, activation='relu', input_dim=4))
        model.add(tf.keras.layers.Dense(units=32, activation='relu'))
        model.add(tf.keras.layers.Dense(units=3, activation='linear'))
        
        return model
    
    def train_model(self):
        self.model = self.neural_network_core()
        
        X_train, X_test, y_train, y_test = self.test_train_split()
        self.model.compile(loss='mean_squared_error', optimizer='adam', 
                           metrics=['accuracy'])
        start_time = time.time()
        
        self.model.fit(X_train, y_train, epochs=1000, batch_size=32, 
                       validation_data=(X_test, y_test))
        
        
        end_time = time.time()
        time_taken = end_time - start_time
        print("Time = {}".format(time_taken))
        
    def predict_df(self):
        
        self.train_model()
        predictions = self.model.predict(self.input_scaled)
        self.predictions = copy.deepcopy((predictions))
        predictions_df = self.inverse_transform()
        self.predictions_df = copy.deepcopy((predictions_df))
        
        return predictions_df
        
    def inverse_transform(self):

        predictions_unscaled = self.scaler.inverse_transform(self.predictions)

        
        predictions_df = pandas.DataFrame(predictions_unscaled, 
                                          columns = self.output_df.columns)
        
        self.visualize_predictions(predictions_df)
        
        return predictions_df
    
    def visualize_predictions(self, predictions_df):

        
        fig, ax = plt.subplots()
        ax.plot(self.input_df['I (A)'], self.output_df['FC Voltage (V)'], 
                color='blue', label='Actual')
        
        ax.plot(self.input_df['I (A)'], predictions_df['FC Voltage (V)'],
                color='red', label='Predictions')
        
        ax.legend()
        plt.show()
        
        
    #def neural_network(self):
        
        
        
# old dynamic response

elif transfer_function == "on":
    
    tf = signal.TransferFunction(num, den)
    
    response_step = response_df.groupby("I_fc")
    
    for response, response_step_df in response_step:

        t1, y1 = signal.step(tf, T = response_step_df["t"].tolist())
        
        y = response_step_df["signal_f"].tolist() * y1
        E = response_step_df["E_OC"].tolist() - y
        E_df = pandas.concat([E_df, pandas.DataFrame(E, columns = ["E"])], ignore_index=True)
        
        # response_df["E"] = pandas.concat([response_df["E"], 
        #                                  pandas.Series(E)], ignore_index=True)



    V_fc = E_df["E"].tolist() - Rohm* I_fc

self.input_df["V_fc"] = V_fc
self.response_df = copy.deepcopy(response_df)

def create_signal():
       
        
    
    # step_number = int((t_end - step_start)/step_duration)

    # df = pandas.DataFrame(columns=["{}".format(signal_name)])

    # for i in range(0,step_number):
        
    #     if i == 0:
    #         t_start = 0
            
    #         if not isinstance(initial_amplitude, type(None)):
                
    #             step_start = t_start
    #             end_time = step_start + step_duration
    #             stepsignal = step_signal(t_start, end_time, step_start, 
    #                                      initial_amplitude)
    #             stepsignal_df = pandas.DataFrame(stepsignal, 
    #                                              columns=["{}".format(signal_name)])
    #             df2 = pandas.concat([df, stepsignal_df], axis=0, 
    #                                 ignore_index = True)
    #             df = copy.deepcopy(df2)
                
    #     else:
    #         step_start = step_start + step_duration
    #         t_start = step_start
    #     signal_amplitude_step = (i+1)*signal_amplitude
            
    #     end_time = step_start + step_duration

    #     stepsignal = step_signal(t_start, end_time, step_start, signal_amplitude_step)
    #     stepsignal_df = pandas.DataFrame(stepsignal, columns=["{}".format(signal_name)])
    #     df2 = pandas.concat([df, stepsignal_df], axis=0, ignore_index = True)
    #     df = copy.deepcopy(df2)

    # t = np.linspace(0, t_end, len(df2))
    
    # df2["t"] = t
    
    # Update required    
    class SimulatedData(ElectroChemicalDynamic):
        
        def __init__(self, electro_model = None, test_mode = False, input_vector = None):
            super().__init__(electro_model, test_mode, input_vector)
            self.input_df = pandas.DataFrame()
            self.output_df = pandas.DataFrame()
            self.Xscaler = None
            self.yscaler = None
            self.X_train_scaled = pandas.DataFrame()
            self.X_test_scaled = pandas.DataFrame()
            self.y_train_scaled = pandas.DataFrame()
            self.y_test_scaled = pandas.DataFrame()
            self.history = pandas.DataFrame()
        
        def find_directory(self):
            # os environment to add to path
            if self.electro_model == "chakraborty":
                directory = "C:/IVI/Fuel_cell/Chakraborty"
                
            elif self.electro_model == "padulles_dynamic1":
                directory = "C:/IVI/Fuel_cell/Padulles-I"
                
            elif self.electro_model == "padulles_dynamic2":
                directory = "C:/IVI/Fuel_cell/Padulles-II"
                
            elif self.electro_model == "padulles_hauer":
                directory = "C:/IVI/Fuel_cell/Padulles-Hauer"
                
            elif self.electro_model == "padulles_amphlett":
                directory = "C:/IVI/Fuel_cell/Padulles-Amphlett"
            
            return directory
            
        def data_explode(self, interval = 5):
            """
            

            Parameters
            ----------
            interval : TYPE, optional
                DESCRIPTION. The default is 5.

            Returns
            -------
            df : TYPE
                DESCRIPTION.

            """
            temperature = np.linspace(333, 353, interval)
            self.select_model()
            df = pandas.DataFrame()
            
            directory = self.find_directory()
                   
            for i in temperature:
                self.input_vector["T"] = i
                self.input_vector["Name"] = "{0}-{1}".format(self.electro_model, i)
                self.select_model()
                file_name = "{}.csv".format(self.input_vector["Name"])
                file_path = os.path.join(directory, file_name)
                df_explode = pandas.read_csv(file_path)
                df_explode["Temperature"] = i
                
                df = pandas.concat([df, df_explode], ignore_index= True)

            return df

        def clean_data(self, data):
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
            data.dropna(inplace= True)
            
            return data
        
        def data_prep(self, interval = 5):
            """
            

            Parameters
            ----------
            interval : TYPE, optional
                DESCRIPTION. The default is 5.

            Returns
            -------
            None.

            """
            data = self.data_explode(interval)
            data_clean = self.clean_data(data)
            
            input_df = data_clean[['I (A)', 'PH2 (atm)', 'PH2O (atm)', 'PO2 (atm)', 
                                   'Temperature'] ]
            
            output_df = data_clean[['FC Voltage (V)', 'FC Efficiency ()', 
                                    'FC Power (W)']]
            
            X_train, X_test, y_train, y_test = self.train_test_split(input_df, 
                                                                     output_df)
            
            X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = self.scaling_data(
                X_train, X_test, y_train, y_test)
            
            self.X_train_scaled = copy.deepcopy(X_train_scaled)
            self.X_test_scaled = copy.deepcopy(X_test_scaled)
            self.y_train_scaled = copy.deepcopy(y_train_scaled)
            self.y_test_scaled = copy.deepcopy(y_test_scaled)
        
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
        
        def scaling_data(self, X_train, X_test, y_train, y_test):
            """
            

            Parameters
            ----------
            X_train : TYPE
                DESCRIPTION.
            X_test : TYPE
                DESCRIPTION.
            y_train : TYPE
                DESCRIPTION.
            y_test : TYPE
                DESCRIPTION.

            Returns
            -------
            X_train_scaled : TYPE
                DESCRIPTION.
            X_test_scaled : TYPE
                DESCRIPTION.
            y_train_scaled : TYPE
                DESCRIPTION.
            y_test_scaled : TYPE
                DESCRIPTION.

            """
            self.Xscaler = StandardScaler()
            self.yscaler = StandardScaler()
            
            X_train_scaled = pandas.DataFrame(self.Xscaler.fit_transform(X_train), 
                                              columns = X_train.columns)
            
            y_train_scaled =  pandas.DataFrame(self.yscaler.fit_transform(y_train), 
                                               columns = y_train.columns)
            
            X_test_scaled = pandas.DataFrame(self.Xscaler.transform(X_test), 
                                             columns = X_test.columns)
            
            y_test_scaled = pandas.DataFrame(self.yscaler.transform(y_test), 
                                             columns = y_test.columns)
            
            return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled
        
        def build_model(self, hp):
            """
            

            Parameters
            ----------
            hp : TYPE
                DESCRIPTION.

            Returns
            -------
            model : TYPE
                DESCRIPTION.

            """
            model = Sequential()
            model.add(Dense(units=hp.Int('units_0', min_value=5, max_value=20, step=5), 
                            activation='relu', input_shape=(self.X_train_scaled.shape[1],)))
            
            num_hidden_layers = hp.Int('num_hidden_layers', min_value=1, max_value=5)
            
            hp_learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4])
            
            
            for i in range(num_hidden_layers):
                model.add(Dense(units=hp.Int(f'units_{i+1}', min_value=5, 
                                             max_value=20, step=5), activation='relu'))
                
            model.add(Dense(units=3, activation='linear'))
            
            model.compile(
                loss='mean_squared_error', 
                optimizer= tf.keras.optimizers.Adam(learning_rate= hp_learning_rate), 
                        metrics=['accuracy'])
            
            return model
        
        def train_model(self):
            """
            

            Returns
            -------
            None.

            """
            self.data_prep()
            tuner = kt.Hyperband(self.build_model, objective= 'val_accuracy',
                                 max_epochs=100, factor=3, directory='kera_tuner', 
                                 project_name='fuel_cell_ann' )
            
            stop_early = tf.keras.callbacks.EarlyStopping(monitor= 'val_loss', 
                                                          patience=3)
            
            tuner.search(self.X_train_scaled, self.y_train_scaled, epochs=10, 
                         validation_data = (self.X_test_scaled, self.y_test_scaled), 
                         callbacks = [stop_early])
      
            best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
            
            model = tuner.hypermodel.build(best_hyperparameters)
            history = model.fit(self.X_train_scaled, self.y_train_scaled, 
                                epochs = 100, validation_data = (self.X_test_scaled,
                                                                 self.y_test_scaled), 
                                callbacks = [stop_early])
            
            self.history = pandas.DataFrame(history.history)
            
            model.save('{}-tuned.h5'.format(self.electro_model))
            
            
        def model_predict(self):
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
            model = tf.keras.models.load_model('{}-tuned.h5'.format(self.electro_model))
            
            predictions = model.predict(self.X_train_scaled)
            self.predictions = copy.deepcopy((predictions))
            
            input_df, output_df, predictions_df = self.inverse_transform()
            
            return input_df, output_df, predictions_df

        def inverse_transform(self):
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
            predictions_unscaled = self.yscaler.inverse_transform(self.predictions)
            input_unscaled = self.Xscaler.inverse_transform(self.X_train_scaled)
            
            predictions_df = pandas.DataFrame(predictions_unscaled,
                                              columns = self.y_train_scaled.columns)
            
            input_df = pandas.DataFrame(input_unscaled, 
                                        columns = self.X_train_scaled.columns)
            
            input_df, output_df, predictions_df = self.visualize_predictions(
                input_df, predictions_df)
            
            return input_df,output_df, predictions_df
        
        def visualize_predictions(self, input_df, predictions_df):
            """
            

            Parameters
            ----------
            input_df : TYPE
                DESCRIPTION.
            predictions_df : TYPE
                DESCRIPTION.

            Returns
            -------
            input_df : TYPE
                DESCRIPTION.
            output_df : TYPE
                DESCRIPTION.
            predictions_df : TYPE
                DESCRIPTION.

            """
            output_unscaled = self.yscaler.inverse_transform(self.y_train_scaled)
            
            output_df = pandas.DataFrame(output_unscaled, 
                                         columns = self.y_train_scaled.columns)
            
            fig, ax = plt.subplots()
            
            ax.plot(input_df['I (A)'], output_df['FC Voltage (V)'], color='blue', 
                    label='Actual')
            
            ax.plot(input_df['I (A)'], predictions_df['FC Voltage (V)'], color='red', 
                    label='Predictions')
            
            ax.legend()
            plt.show()
            
            return input_df, output_df, predictions_df
        
        
        
     # #x1   
     # def oxygen_flow_rate(self):
     #     mO2_t = (1/lamb_c)*(I/4*F - mO2)
     #     return mO2_t
     
     # #x2
     # def hydrogen_flow_rate(self):
     #     mH2_t = (1/lamb_a)*(I/2*F - mH2)
     #     return mH2_t
     
     # #x3
     # def h2o_flow_rate(self):
     #     mH2O_t = (1/lamb_c) * (I/2*F - mH2Onet)
     #     return mH2O_t
                   
     # #x4 x5
     # def hydrogen_pressure(self):
     #     pH2_t = 2*((R*(mH2Oa_in)*T)/Va*(PH2Oa_in))*u_pa - 2*((R*(mH2Oa_in)*T)/Va*(PH2Oa_in))*PH2 - ((R*T)/(4*Vc*F))*I
     #     return pH2_t
     
     # #x6
     # def oxygen_pressure(self):
     #     pO2_t = 2* ((R*(mH2Oc_in)*T)/Vc*(PH2Oc_in))*u_pc - 2*((R*(mH2Oc_in)*T)/Vc*(PH2Oc_in)*PO2 -  ((R*T)/(4*Vc*F)))*I
         
     #     return pO2_t
     
     # def h2o_pressure(self):
         
         def calc_theta_1(self, state="current"):
             cell_parameters = copy.deepcopy(self.cell_parameters)
             R = cell_parameters["R"]
             Va = cell_parameters["Va"]
             mH2Oa_in = cell_parameters["mH2O"]
             PH2Oa_in = cell_parameters["PH2O_e"]
             
             if state == "previous":
                 x4 = self.previous_state_variables["T"]
                 
             elif state=="current":
                 x4 = self.current_state_variables["T"]
                 
             else:
                 raise ValueError("Invalid state = {}".format(state))
                 
             theta_1 = (R*mH2Oa_in*x4)/(Va*PH2Oa_in)
             
             return theta_1
         
         def calc_theta_2(self, state = "current"):
             
             cell_parameters = copy.deepcopy(self.cell_parameters)
             mH2Oa_in = cell_parameters["mH2O"]
             R = cell_parameters["R"]
             x4 = self.current_state_variables["T"]
             theta_2 = R*mH2Oa_in*x4
             return theta_2
             
         

         def calc_theta_3(self, state = "current"):
             cell_parameters = copy.deepcopy(self.cell_parameters)
             R = cell_parameters["R"]
             Vc = cell_parameters["Vc"]
             mH2Oc_in = cell_parameters["mH2O"]
             PH2Oc_in = cell_parameters["PH2O_e"]
             
             if state == "previous":
                 x4 = self.previous_state_variables["T"]
                 
             elif state=="current":
                 x4 = self.current_state_variables["T"]
                 
             else:
                 raise ValueError("Invalid state = {}".format(state))
                 
             theta_3 = (R*mH2Oc_in*x4)/(Vc*PH2Oc_in)
             
             return theta_3
         
         def calc_theta_4(self, state = "current"):
             cell_parameters = copy.deepcopy(self.cell_parameters)
             R = cell_parameters["R"]
             F = cell_parameters["F"]
             Vc = cell_parameters["Vc"]
             x4 = self.current_state_variables["T"]
             theta_4 = (R*x4)/(4*Vc*F)
             
             return theta_4
         
         def calc_theta_5(self, state = "current"):
             cell_parameters = copy.deepcopy(self.cell_parameters)
             R = cell_parameters["R"]
             Vc = cell_parameters["Vc"]
             mH2Oc_in = cell_parameters["mH2O"]
             PH2Oc_in = cell_parameters["PH2O_e"]
             PH2O_in = cell_parameters["PH2O"]
             if state == "previous":
                 x7 = self.previous_state_variables["PH2O"]
                 
             elif state == "current":
                 x7 = self.current_state_variables["PH2O"]
                 
             else:
                 raise ValueError("Invalid state = {}".format(state))
             theta_5 = (R* mH2Oc_in *(PH2O_in - x7))/Vc*(PH2Oc_in)
             
             return theta_5
         
         def calc_theta_6(self, state = "current"):
             current_state_variables = copy.deepcopy(self.current_state_varibles)
         
            
class StateSpaceModel_old:
    def __init__(self, initial_state_variables = None):
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
                                "m_H2O": 8.614 * 10**(-5),
                                "ns": 48,
                                "PH2O": 2,
                                "R": 8.31,
                                "R0_c": 0.28,
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
        
        if not isinstance(initial_state_variables, type(None)):
            self.previous_state_variables = copy.deepcopy(initial_state_variables)
            
            
        
    @property
    def matrix_a_44(self):
        cell_parameters = copy.deepcopy(self.cell_parameters)
        hs = cell_parameters["hs"]
        ns = cell_parameters["ns"]
        A_s = cell_parameters["A_s"]
        M_fc = cell_parameters["M_fc"]
        Cfc = cell_parameters["Cfc"]
        
        a_44 = (-hs *ns * A_s)/(M_fc * Cfc)
        return a_44
    
    def matrix_a_11(self, I):
        cell_parameters = copy.deepcopy(self.cell_parameters)

        C = cell_parameters["C"]
        Vact = self.activation_loss()
        Rconc = self.concentration_resistance()
        Ract = Vact/I
        
        a_11 = (-1)/ C*(Ract + Rconc)
        
        return a_11

    def activation_loss(self, I):
        
        cell_parameters = copy.deepcopy(self.cell_parameters)
        a = cell_parameters["a"]
        b = cell_parameters["b"]
        a0 = cell_parameters["a0"]
        T = self.state_variables["T"]
        Vact = a0 + T * (a + (b*np.log(I)))
        
        return Vact
    
    # Valid only for Avista labs SR-12
    def concentration_resistance(self, I):
        Rconc0 = 0.080312
        Rconc1 = 5.2211*(10**(-8))*(I**6) - 3.4578*(10**(-6))*(I**5) + 8.6437*(10**(-5))*(I**4) - 0.010089**(I**3) + 0.005554*(I**2) - 0.010542*I
        T = self.state_variables["T"]
        Rconc2 = 0.0002747*(T-298)
        Rconc = Rconc0 + Rconc1 + Rconc2
        
        return Rconc
    
    def ohmic_loss(self, T, I):
        
        cell_parameters = copy.deepcopy(self.cell_parameters)
        Roc = cell_parameters["Roc"]
        K_I = cell_parameters["K_I"]
        K_T = cell_parameters["K_T"]
        Ro = Roc + K_I*I + K_T * T
        
        return Ro
   
    def calc_theta(self, I):
        
        current_state_variables = self.current_state_variables
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
        theta_2 = R*mH2Oa_in*x4
        
        Vc = cell_parameters["Vc"]
        mH2Oc_in = cell_parameters["mH2O"]
        PH2Oc_in = cell_parameters["PH2O_e"]
        theta_3 = (R*mH2Oc_in*x4)/(Vc*PH2Oc_in)
        theta_4 = (R*x4)/(4*Vc*F)
        
        x7 = current_state_variables["PH2O"]
        theta_5 = (R* mH2Oc_in *(PH2O_in - x7))/Vc*(PH2Oc_in)
        theta_6 = (ns*del_G0)/(2*F) - (ns*R*x4/2*F)*np.log(x5*(x6**0.5)/x7)
        
        Rconc = self.concentration_loss(x4, I)
        Ro = self.ohmic_loss(x4, I)
        Vo = Ro * I
        Vconc = Rconc*I
        Vact = self.activation_loss(I)
        E0_cell = cell_parameters["E0_cell"]
        
        theta_7 = ns*(E0_cell + (R*x4)/(2*F) * np.log(x5*(x6**0.5)/x7) - Vact - Vconc - Vo)
        Mfc = cell_parameters["M_fc"]
        Cfc = cell_parameters["Cfc"]
        theta_8 = ns*((2*E0_cell/Mfc*Cfc) + R*x4/F*Mfc*Cfc)*np.log(x5*x6)
        self.current_theta = {"theta_1": theta_1,
                      "theta_2": theta_2,
                      "theta_3": theta_3,
                      "theta_4": theta_4,
                      "theta_5": theta_5,
                      "theta_6": theta_6,
                      "theta_7": theta_7,
                      "theta_8": theta_8}   
        
        
        
        
    def matrix_a(self, I, state = "current"):
        
        cell_parameters = copy.deepcopy(self.cell_parameters)
        hs = cell_parameters["hs"]
        ns = cell_parameters["ns"]
        As = cell_parameters["As"]
        a_matrix = np.zeros((11,11))
        a_44 = self.matrix_a_44
        a_11 = self.matrix_a_11(I)
        cell_parameters = copy.deepcopy(self.cell_parameters)
        theta_1 = self.calc_theta_1(state)
        theta_5 =  self.calc_theta_5(state)
        theta_3 = self.calc_theta_3(state)
        lamb_a = cell_parameters["lamb_a"]
        lamb_c = cell_parameters["lamb_c"]
      
        for index_i, row in a_matrix:
            for index_j, element in row:
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
    
    def matrix_b(self, state = "current"):
        b_matrix = np.zeros((11,3))
        cell_parameters = copy.deepcopy(self.cell_parameters)
        I = self.current_input_state["I"]
        
        b_12 = -1 * self.matrix_a_44
        current_theta = self.calc_theta(current_state_variables, I)
        theta_1 = self.calc_theta_1(state)
        theta_3 = self.calc_theta_3(state)
        hs = cell_parameters["hs"]
        ns = cell_parameters["ns"]
        As = cell_parameters["As"]
        
        
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
        cell_paramaters = copy.deepcopy(self.cell_parameters)
        lamb_c = cell_parameters["lamb_c"]
        lamb_a = cell_parameters["lamb_a"]
        F = cell_paramaters["F"]
        
    # def model(self, y):

        

