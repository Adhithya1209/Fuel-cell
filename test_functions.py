# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 15:37:58 2023

@author: jayaraman
"""
#%% test pca

from pca import PCA_Components

pca_an = PCA_Components()
data =pca_an.prepare_data()
pca_result = pca_an.pca_analysis("StandardScaler")
pca_an.plot_variance_ratio()
unscaled_data = pca_an.pca_inverse()
i_fc = unscaled_data[:120,0]
V_fc = unscaled_data[0, 1:121]
pca_an.plot(i_fc, V_fc, "current", "voltage", "Polarisation curve")
