"""
RestNet model classes.
Classes
-------
    WASS_loss  : Wasserstein distance loss

@author: Cecile Della Valle
@date: 03/01/2021
"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
# General import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import cv2 as cv
import os
from PIL import Image
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`




def CreateDataSet(test,path, noise=0.1, save='yes'):
    # Recuperation des donnees
    nx            = test.nx
    # Initialisation
    color         = ('b','g','r')
    liste_lisse   = []
    liste_l_trsf  = []
    liste_blurred = []
    liste_b_trsf  = []
    # Upload Data
    # path : './Datasets/BSD500_COCO1000_train_val/train/'
    for folder, subfolders, filenames in os.walk(path): 
        for img in filenames: 
            item   = folder+img
            img_cv = cv.imread(item,cv.IMREAD_COLOR)
            for i,col in enumerate(color):
                # Etape 1 : obtenir l'histogramme lisse des couleurs images
                histr = cv.calcHist([img_cv],[i],None,[256],[0,256]).squeeze()
                # Savitzky-Golay
                y     = savgol_filter(histr, 21, 5)
                # interpolation pour nx points
                x     = np.linspace(0,1,256, endpoint=True)
                xp    = np.linspace(0,1,nx,endpoint=True)
                f     = interp1d(x,y)
                yp    = f(xp)
                # normalisation
                x_true     = yp/np.linalg.norm(yp)
                # save
                liste_lisse.append(x_true)
                # Etape 2 : passage dans la base de T^*T
                x_true_trsf = test.BaseChange(x_true)
                liste_l_trsf.append(x_true_trsf)
                #  Etape 3 : obtenir les images bruitees par l' operateur d' ordre a
                # transform and add noise
                x_b_trsf    = test.Compute(x_true_trsf) 
                # Etape 4 : Retour dans la base des elements finis
                x_blurred   = test.BaseChangeInv(x_b_trsf)
                # Etape 5 : Bruitage 
                noise_vect  = np.random.randn(nx)
                noise_vect  = noise_vect/np.linalg.norm(noise_vect)
                x_blurred  += noise*noise_vect
                # save
                liste_blurred.append(x_blurred)
                liste_b_trsf.append(test.BaseChange(x_blurred))
    # Export data in .csv
    if save =='yes':
        np.savetxt('data_lisse.csv',      liste_lisse,   delimiter=', ', fmt='%12.8f')
        np.savetxt('data_lisse_trsf.csv', liste_l_trsf,  delimiter=', ', fmt='%12.8f')
        np.savetxt('data_blurred.csv',    liste_blurred, delimiter=', ', fmt='%12.8f')
        np.savetxt('data_blurred_trsf.csv',liste_b_trsf, delimiter=', ', fmt='%12.8f')
    # Tensor completion
    x_tensor = np.array(liste_l_trsf)
    y_tensor = np.array(liste_b_trsf)
    #
    dataset = TensorDataset(x_tensor, y_tensor)
    l       = len(dataset)
    ratio   = 2*l//3
    train_dataset, val_dataset = random_split(dataset, [ratio, l-ratio])
    #
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False)
    #
    return train_loader, val_loader

def CreateLoader(folder,nsample):
    """
    According to the mode, creates the appropriate loader 
    for the training and validation sets.
    To reuse a data set.
    """
    dfx     = pd.read_csv(folder+'/'+'data_lisse.csv', sep=',',header=None)
    dfy     = pd.read_csv(folder+'/'+'data_blurred.csv', sep=',',header=None)
    _,m    = dfx.shape
    #
    x_tensor= torch.FloatTensor(dfx.values[:nsample]).view(-1,1,m)
    y_tensor= torch.FloatTensor(dfy.values[:nsample]).view(-1,1,m)
    #
    dataset = TensorDataset(x_tensor, y_tensor)
    l = len(dataset)
    ratio = 2*l//3
    train_dataset, val_dataset = random_split(dataset, [ratio, l-ratio])
    #
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False)
    return train_loader, val_loader
    
            
