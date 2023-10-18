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
         
        

