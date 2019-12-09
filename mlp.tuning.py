# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 15:49:12 2019
MLP hyperparameter tuning
@author: mi292519
"""
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import time
from autocorrplot import autcorrplt
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from keras import regularizers


def mlp_tune():
    
    def create_model(layers, activation, optimizer):
        model = Sequential()
        for i, nodes in enumerate(layers):
            if i==0:
                model.add(Dense(nodes,input_dim=x_norm_train.shape[1]))
                model.add(Activation(activation))
            else:
                model.add(Dense(nodes))
                model.add(Activation(activation))
        model.add(Dense(1, activation = 'linear')) # Note: no activation beyond this point
        
        model.compile(optimizer= optimizer, loss='mse')
        # optimizers.Adam(learning_rate= rate, beta_1=0.9, \
        #                       beta_2=0.999, amsgrad=False)
        return model
    
    model = KerasRegressor(build_fn=create_model, verbose=1)

    #specifying layer architecture
    optimizer = ['adam', 'rmsprop', 'sgd','adagrad', 'adadelta'] 
    layers = [[3], [10], [30], [10, 10], [10, 20], [20, 20], \
              [30, 30], [10, 10, 10], [20,20,20], \
                  [30,30,30], [10,20,30], [20,20,30]]
    activations = ['relu', 'tanh', 'sigmoid']
    param_grid = dict(layers=layers, optimizer = optimizer, activation=activations, \
                      batch_size = [10, 50, 100], epochs=[10, 50])
    grid = GridSearchCV(estimator=model, param_grid=param_grid,\
                        scoring='neg_mean_squared_error')
    
    grid_result = grid.fit(x_norm_train, y_train)
    
    [grid_result.best_score_, grid_result.best_params_]
    
    testPredict = grid.predict(x_norm_test)
    
    #make model evaluation
    from sklearn.metrics import mean_squared_error
    print()
    print(mean_squared_error(y_test, testPredict))
    print()
   
    #list all data in history
    print(history.history.keys())
    
    # summarize history for accuracy
    plt.figure(figsize=(15,8))
    plt.plot(grid_result.history['mean_squared_error'])
    plt.plot(grid_result.history['val_mean_squared_error'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    
    # summarize history for loss
    plt.figure(figsize = (10,8))
    plt.plot(grid_result.history['loss'])
    plt.plot(grid_result.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    #prepare data for plotting
    yy = y_test[:]
    yy.reset_index(inplace=True)
    yy.drop(['index'], axis = 1, inplace=True) 
    
    
    #plotting 
    %matplotlib qt5
    sns.set_context('notebook', font_scale= 1.5)
    plt.figure(figsize=(15,8))
    plt.plot(yy['surge'])
    plt.plot(testPredict, color= 'red')
    plt.legend(['Observed Surge', 'Modeled Surge'],fontsize = 14)
    plt.ylabel('Surge Height (m)')

    