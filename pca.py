# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 05:55:09 2023

@author: jayaraman
"""

import pandas
import numpy as np
import matplotlib.pyplot as plt
import copy
import ast
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
#Inherit Preprocessdata
class PCA_Components():
    def __init__(self, data = None, file='state_space_vis.csv'):
        self.is_data = None
        self.scaler=None
        self.pca = None
        if isinstance(data, type(None)):
            self.data = copy.deepcopy(data)
        
        if isinstance(data, type(None)):
            self.file = file
        
    def prepare_data(self, file=None):
        
        
        if not isinstance(file, type(None)):
            self.file = file
        
        try:
            data = pandas.read_csv(self.file, index_col=0)
            
        except ValueError:
            raise ValueError("Invalid file name")
            
        
        data['V_fc'] = data['V_fc'].apply(ast.literal_eval)
        data['I_fc'] = data['I_fc'].apply(ast.literal_eval)
        df_new = pandas.DataFrame(data['V_fc'].tolist(), index=data.index)
        df_result = pandas.concat([data["I_fc"], df_new], axis=1)
        df_result = df_result.explode("I_fc")
        df_result.reset_index(drop=True,inplace=True)
        columns = df_result.columns[1:121]
        df_result.rename(columns=dict(zip(columns, [f'v{i}' for i in range(1, 121)])), inplace=True)
        self.data = copy.deepcopy(df_result)
        return df_result
    
    def scale_data(self, scaler="MinMaxScaler"):
        if scaler=="MinMaxScaler":
            scaler = MinMaxScaler()
            
        elif scaler=="StandardScaler":
            scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data)
        self.scaler = copy.deepcopy(scaler)
        return scaled_data
        
    def pca_analysis(self, scaler="MinMaxScaler"):
        pca = PCA(n_components=10)
        scaled_data = self.scale_data(scaler)
        pca_result = pca.fit_transform(scaled_data)
        self.pca = copy.deepcopy(pca)
        return pca_result
    
    def pca_post(self):
        pca = PCA(2)
        scaled_data = self.scale_data()
        pca_result = pca.fit_transform(scaled_data)
        
        return pca_result, pca
        
        
    def pca_inverse(self):
        pca_result, pca = self.pca_post()
        scaled_data = pca.inverse_transform(pca_result)
        
        data_unscaled = self.inverse_scaler(scaled_data)
        return data_unscaled
    
    def inverse_scaler(self, scaled_data):
        
        data_unscaled = self.scaler.inverse_transform(scaled_data)
        
        return data_unscaled
    
    def plot(self,x, y,xlabel="", ylabel="", title=""):
        plt.plot(x, y)
        plt.xlabel('{}'.format(xlabel))
        plt.ylabel('{}'.format(ylabel))
        plt.title('{}'.format(title))
        plt.grid()
        plt.show()
    
    def plot_variance_ratio(self):
        explained_variance_ratio = self.pca.explained_variance_ratio_
        pc_number = np.arange(self.pca.n_components_) + 1
        plt.plot(pc_number, explained_variance_ratio, 'ro-')
        plt.xlabel('Number of components')
        plt.ylabel('Proportion of Variance')
        plt.title('Elbow Plot for PCA')
        plt.grid()
        plt.show()
