# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 05:59:35 2023

@author: jayaraman
"""

import fuel_cell
import pandas
import matplotlib.pyplot as plt
import numpy as np
import ast

#%% Simulate nexa - RL model
cell_parameters = {"Eoc": 30.3,
                        "R1" : 0.154,
                        "R2" : 0.1515,
                        "L": 4.94}
t_start = 0
t_end = 350
step_start = 65
step_duration = 285
signal_amplitude = 33
signal_name = "I_fc"
initial_amplitude = 13

df_test = fuel_cell.create_signal(t_start, t_end, step_start, step_duration, signal_amplitude, signal_name, initial_amplitude, plot = True)
nexa = fuel_cell.EquivalentCircuitRL(input_signal = df_test, cell_parameters = cell_parameters, test=True)
V_fc_df = nexa.output_voltage()
x = nexa.output_signal["t"]
y = nexa.output_signal["V_fc"]
#nexa.plot_results(x,y,"time","V_fc","output_signal")

#%% Experimental - Enguage Digitizer

data_exp = pandas.read_csv("newequivalentcircuit-enguage.csv")
new_column_names = {
    'x': 't',
    'Curve1': 'V_fc'
}

data_exp.rename(columns = new_column_names, inplace = True)
i_fc = []

for t in data_exp["t"]:
    for index,t2 in enumerate(nexa.output_signal["t"]):
        if index == len(nexa.output_signal["t"])-1 and t>nexa.output_signal["t"].iloc[index]:
            
            i_fc.append(nexa.output_signal["I_fc"].iloc[index])
            
        elif t >= nexa.output_signal["t"].iloc[index] and t< nexa.output_signal["t"].iloc[index+1]:
            i_fc.append(nexa.output_signal["I_fc"].iloc[index])
            
            
data_exp["I_fc"] = i_fc
#%% Train LSTM without noise
Data_prep = fuel_cell.PreprocessData(data=data_exp)
Data_prep.data_prep()
Data_prep.initialise_tensor_data()
lstm_model = fuel_cell.LSTM_model()

lstm_model.train_model()
predictions = lstm_model.model_predict()

predictions_unscaled = Data_prep.inverse_transform(predictions)
t=data_exp["t"][:72]
fuel_cell.plot_results(t, predictions_unscaled, "time", "Voltage", "output_signal")
#lstm_model.visualize_loss()

# get time values for X_test
#nexa.plot_results(x,y,"time","V_fc","output_signal")
# fuel_cell.plot_results()

#%% Merge experimental and RL

data = pandas.concat([data_exp, nexa.output_signal], ignore_index=True)

data_sorted = data.sort_values(by ="t", ascending=True)
#%% Merge accuracy without noise
Data_prep = fuel_cell.PreprocessData(data=data_sorted)
Data_prep.data_prep()
Data_prep.initialise_tensor_data()
lstm_model = fuel_cell.LSTM_model()

lstm_model.train_model()
predictions = lstm_model.model_predict()

predictions_unscaled = Data_prep.inverse_transform(predictions)

t=data_exp["t"][:72]
fuel_cell.plot_results(t, predictions_unscaled, "time", "Voltage", "output_signal")
# get time values for X_test
#nexa.plot_results(x,y,"time","V_fc","output_signal")
# fuel_cell.plot_results()

#%% LSTM with noise
Data_prep = fuel_cell.PreprocessData(data=data_exp)
Data_prep.data_prep()
Data_prep.initialise_tensor_data()
lstm_model = fuel_cell.LSTM_model(noise=True)

lstm_model.train_model()
predictions = lstm_model.model_predict()

predictions_unscaled = Data_prep.inverse_transform(predictions)

t=data_exp["t"][:72]
fuel_cell.plot_results(t, predictions_unscaled, "time", "Voltage", "output_signal")
# get time values for X_test
#nexa.plot_results(x,y,"time","V_fc","output_signal")
# fuel_cell.plot_results()
#%% Merge accuracy with noise
Data_prep = fuel_cell.PreprocessData(data=data_sorted)
Data_prep.data_prep()
Data_prep.initialise_tensor_data()
lstm_model = fuel_cell.LSTM_model(noise=True)

lstm_model.train_model()
predictions = lstm_model.model_predict()

predictions_unscaled = Data_prep.inverse_transform(predictions)
t=data["t"][:28072]
fuel_cell.plot_results(t, predictions_unscaled, "time", "Voltage", "output_signal")

# get time values for X_test
#nexa.plot_results(x,y,"time","V_fc","output_signal")
# fuel_cell.plot_results()
#%% GMM - ANN
Data_prep = fuel_cell.PreprocessData(fuelcell_model="GenericMatlabModel_aug")

Data_prep.data_prep_regression()
Data_prep.initialise_tensor_data()
ann_model = fuel_cell.ANN_model(learning_rate=1, noise=False)
ann_model.train_model(model_name="GMM_ANN_untuned", X_train=Data_prep.X_train_scaled, y_train=Data_prep.y_train_scaled)
predictions = ann_model.model_predict(model_name="GMM_ANN_untuned", X_train=Data_prep.X_train_scaled)
predictions_unscaled = Data_prep.inverse_transform_regression(predictions)
# data=Data_prep.build_data(X_train=Data_prep.X_train_scaled, predictions=predictions)
# data_unscaled= Data_prep.inverse_transform_regression(predictions=predictions, data=data)
#%% ANN prediction of GMM database
#df= pandas.DataFrame()
df= pandas.read_csv("GMM_nedstack-vis.csv", index_col=0)
df['V_fc'] = df['V_fc'].apply(ast.literal_eval)
df['I_fc'] = df['I_fc'].apply(ast.literal_eval)
data_prep = fuel_cell.PreprocessData(data=df)
df.rename(columns={'V_fc': 'A', 'I_fc': 'C'}, inplace=True)
df = df.explode(list('AC'))
df.rename(columns={'A': 'V_fc', 'C': 'I_fc'}, inplace=True)
df.reset_index(drop=True,inplace=True)
scaled_data =data_prep.scale_data(df)
X = np.hstack((scaled_data[:, :3], scaled_data[:, 4:]))
y = scaled_data[:, 3]
ann_model = fuel_cell.ANN_model()
predictions=ann_model.model_predict(X[:120,:], model_name="GMM_ANN_untuned")
predictions_unscaled =data_prep.inverse_transform_regression(predictions)[:,:,3]
I_fc = df["I_fc"].head(120)
plt.plot(df["I_fc"].head(120), predictions_unscaled)

#%% SSM - ANN
""" simulation yet to be done"""
#Data_prep = fuel_cell.PreprocessData(fuelcell_model="StateSpaceModel_aug")
Data_prep = fuel_cell.PreprocessData()
Data_prep.data_prep_regression()
Data_prep.initialise_tensor_data()
ann_model = fuel_cell.ANN_model(learning_rate=1, noise=False)
ann_model.train_model(model_name="GMM_ANN_untuned", X_train=Data_prep.X_train_scaled, y_train=Data_prep.y_train_scaled)
predictions = ann_model.model_predict(model_name="GMM_ANN_untuned", X_train=Data_prep.X_train_scaled)
predictions_unscaled = Data_prep.inverse_transform_regression(predictions)

#%% ANN Prediction of SSM database

df= pandas.read_csv("state_space_vis.csv", index_col=0)
df['V_fc'] = df['V_fc'].apply(ast.literal_eval)
df['I_fc'] = df['I_fc'].apply(ast.literal_eval)
data_prep = fuel_cell.PreprocessData(data=df)
df.rename(columns={'V_fc': 'A', 'I_fc': 'C'}, inplace=True)
df = df.explode(list('AC'))
df.rename(columns={'A': 'V_fc', 'C': 'I_fc'}, inplace=True)
df.reset_index(drop=True,inplace=True)
scaled_data =data_prep.scale_data(df)
X = np.hstack((scaled_data[:, :3], scaled_data[:, 4:]))
y = scaled_data[:, 3]
ann_model = fuel_cell.ANN_model()
predictions=ann_model.model_predict(X[:120,:], model_name="GMM_ANN_untuned")
predictions_unscaled =data_prep.inverse_transform_regression(predictions)[:,:,3]
I_fc = df["I_fc"].head(120)
plt.plot(df["I_fc"].head(120), predictions_unscaled)