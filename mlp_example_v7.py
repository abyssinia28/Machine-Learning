# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 15:05:53 2019
MLP implementation - make a function for it!
@author: mi292519
"""
#mc.ai example
import os
os.chdir('F:\\OneDrive - Knights - University of Central Florida\\UCF\Projekt.28\\Coursework\Fall_2019\\Machine_Learning\\Project\\Script\ml_project')
from timelag import time_lag
from mlp_sequential import mlp_seq
from lstm import lstm
from pca import pca
from sklearn import preprocessing
#%matplotlib qt5



#Data preprocessing
data = '2011_cux.csv'
x, surge = time_lag(data,5)




 

#Splitting to training and testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
x, surge, shuffle=False, test_size=0.25, random_state=42)

#standardizing
x_norm_train = preprocessing.scale(x_train)
x_norm_test = preprocessing.scale(x_test)

# #PCA
# x_train = pca(x_norm_train, x_norm_test)[0]
# x_test = pca(x_norm_train, x_norm_test)[1]


#MLP
mlp_seq(x_norm_train, x_norm_test, y_train, y_test)


#LSTM
lstm(x_norm_train, x_norm_test, y_train, y_test)

