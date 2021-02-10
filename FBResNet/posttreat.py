"""
RestNet model classes.
Modules
-------
    Export_Data  : ...

@author: Cecile Della Valle
@date: 03/01/2021
"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
# General import
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
import os
import sys
from PIL import Image
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`


### EXPORT DATA
def Export_Data(xdata,ydata,folder,name):
    """
    Save a function in a chose folder
    for plot purpose.
    """
    Npoint = np.size(xdata)
    with open(folder+'/'+name+'.txt', 'w') as f:
        f.writelines('xdata ydata \n')
        for i in range(Npoint):
            web_browsers = ['{0}'.format(xdata[i]),' ','{0} \n'.format(ydata[i])]
            f.writelines(web_browsers)

### PLOT GAMMA ALPHA MU
def Export_hyper(resnet,x,x_b,folder):
    nlayer = len(resnet.model.Layers)
    gamma  = np.zeros(nlayer)
    reg    = np.zeros(nlayer)
    mu     = np.zeros(nlayer)
    for i in range(0,nlayer):
        gamma[i] = resnet.model.Layers[i].gamma_reg[0]
        reg[i]   = resnet.model.Layers[i].gamma_reg[1]
        mu[i]    = resnet.model.Layers[i].mu
    # export
    num    = np.linspace(0,nlayer-1,nlayer)
    Export_Data(num, gamma, folder, 'gradstep')
    Export_Data(num, reg, folder, 'reg')
    Export_Data(num, mu, folder, 'prox')
    # plot
    fig, (ax0,ax1,ax2) = plt.subplots(1, 3)
    ax0.plot(num,gamma)
    ax0.set_title('gradstep')
    ax1.plot(num,reg)
    ax1.set_title('reg')
    ax2.plot(num,mu)
    ax2.set_title('prox')
    plt.show()
    
    