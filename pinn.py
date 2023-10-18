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
        
        
        
# old dyn