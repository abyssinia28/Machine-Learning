# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 14:24:17 2019
Autocorrelation plot
@author: mi292519
"""

import matplotlib.pyplot as plt 


def autcorrplt(data, lag, *args):
    """
    plots the autocorrelation of a pandas series object
    """
    acorr = [];
    for ii in range(lag+1):
        acorr.append(data.autocorr(lag = ii))
    
    plt.figure(figsize = (8,6))
    plt.plot(acorr)
    plt.xlabel("Lag (in 6 hrs)", fontsize = 20)
    plt.ylabel("Correlation", fontsize = 20)
    plt.ylim([0,1])
    plt.xlim([1,lag])
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.title("Autocorrelation in slp", fontsize = 20)
    
   

    


    
