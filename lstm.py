# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 14:55:13 2019
LSTM
@author: mi292519
"""
import math
import time
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from keras import regularizers


def lstm(x_norm_train, x_norm_test, y_train, y_test):
    #defining model
    model = Sequential()
    model.add(LSTM(94, activation='relu', input_shape = (1, x_norm_train.shape[1])))
    model.add(Dense(1))
    model.compile(loss = 'mse', optimizer = 'adam')
    
    #reshaping data to 3D
    x_train_resh = x_norm_train.reshape(x_norm_train.shape[0], \
                                        1, x_norm_train.shape[1])
    x_test_resh = x_norm_test.reshape(x_norm_test.shape[0], \
                                      1, x_norm_test.shape[1])
    
    #fit the model
    history = model.fit(x_train_resh, y_train['surge'], epochs = 50, \
                            batch_size = 20, verbose = 1, validation_split=0.2)
    
    plt.plot(history.history['loss'], label = 'train')
    plt.plot(history.history['val_loss'], label = 'test')
    plt.legend()
    
    res = model.predict(x_test_resh)
    
    #prepare data for plotting
    yy = y_test[:]
    yy.reset_index(inplace=True)
    yy.drop(['index'], axis = 1, inplace=True) 
    
    #make model evaluation
    from sklearn.metrics import mean_squared_error, \
        mean_absolute_error, r2_score
    print()
    print("mse = ", mean_squared_error(y_test['surge'], res))
    print("mae = ", mean_absolute_error(y_test['surge'], res))
    print("r2_score = ", r2_score(y_test['surge'], res))

    print()
    
    #plotting 
     #plotting 
    sns.set_context('notebook', font_scale= 1.5)
    plt.figure(figsize=(20,6))
    plt.plot(y_test['date'], yy['surge'], color = 'blue')
    plt.plot(y_test['date'],res, color= 'red')
    plt.legend(['Observed Surge', 'Modeled Surge'],fontsize = 14)
    plt.ylabel('Surge Height (m)')
 
    # summarize history for loss
    plt.figure(figsize = (15,8))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    return res