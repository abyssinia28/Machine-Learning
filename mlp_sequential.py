# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 15:19:05 2019
MLP Sequential Model Implementation
@author: mi292519
"""
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import math
import time
from autocorrplot import autcorrplt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from keras import regularizers
from keras.layers import Dropout, BatchNormalization



def mlp_seq(x_norm_train, x_norm_test, y_train, y_test):
    """
    Builds mlp, trains and tests it
    """
    #simple model
    model = Sequential()
    model.add(Dropout(0.2, input_shape = (x_norm_train.shape[1],)))
    model.add(Dense(94, activation = 'sigmoid',\
                    input_shape = (x_norm_train.shape[1],)))
    model.add(Dense(94, activation='sigmoid'))
    model.add(BatchNormalization())
    # model.add(Dense(180, activation='relu'))
    model.add(Dense(1))
    
    #Training model
    model.compile(loss = 'mean_squared_error', optimizer = 'adagrad', \
                  metrics=['mean_squared_error', 'mean_absolute_error'])
    history = model.fit(x_norm_train, y_train['surge'], epochs = 50, \
              batch_size = 10, verbose = 1, validation_split=0.2)
    
    testPredict = model.predict(x_norm_test)
    
    #prepare data for plotting
    yy = y_test[:]
    yy.reset_index(inplace=True)
    yy.drop(['index'], axis = 1, inplace=True) 
    
    #make model evaluation
    from sklearn.metrics import mean_squared_error, \
        mean_absolute_error, r2_score
    print()
    print("mse = ", mean_squared_error(y_test['surge'], testPredict))
    print("mae = ", mean_absolute_error(y_test['surge'], testPredict))
    print("r2_score = ", r2_score(y_test['surge'], testPredict))

    print()
    
    #plotting 
    sns.set_context('notebook', font_scale= 1.5)
    plt.figure(figsize=(20,6))
    plt.plot(y_test['date'], yy['surge'], color = 'blue')
    plt.plot(y_test['date'],testPredict, color= 'red')
    plt.legend(['Observed Surge', 'Modeled Surge'],fontsize = 14)
    plt.ylabel('Surge Height (m)')

    fig, ax = plt.subplots()
    ax.scatter(y_test['surge'], testPredict, c='black')
    line = mlines.Line2D([0, 1], [0, 1], color='red')
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)
    plt.show()

    #list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.figure(figsize=(15,8))
    plt.plot(history.history['mean_squared_error'])
    plt.plot(history.history['val_mean_squared_error'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.figure(figsize = (15,8))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    return testPredict