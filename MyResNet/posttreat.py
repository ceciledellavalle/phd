"""
RestNet model classes.
Classes
-------
    Lifschitz : Lifschitz constant of the network

@author: Cecile Della Valle
@date: 03/01/2021
"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
# General import
import torch
import torch.nn as nn
import numpy as np
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`

def Lifschitz(model,nx,nL,a=1,p=1):
    """
    Given a ill-posed problem of order a and a regularization of order p
    for a 1D signal of nx points,
    the fonction Physics compute the tensor of the linear transformation Trsf
    and the tensor used in the algorithm.
    Parameters
    ----------
        model (MyModel): model of the neural network
        nx        (int):
        nL        (int): number of layers
        a        (int) : order of ill-posedness (default 1)
    Returns
    -------
        (float): Lifschitz constant theta of the neural network
    
    """
    # Etape 0 : vectors of eigenvals of T^*T and D^*D
    eig_T = (np.linspace(0,nx-1,nx)-1/2)*np.pi
    eig_D = 1/eig_T**p
    # Etape 1 : on parcourt les layers du reseau
    for i in range(0,len(self.Layers)):
        # Acces to the parameters of each layers
        gamma = model.Layers[i].gamma.numpy()
        reg   = model.Layers[i].reg.numpy()
        mu    = 1
        # Computes the ref eigenvals
        eig_ref[i]    = 1 - gamma*(eig_T+reg*eig_D)
        eig_ipn[:i]   = eig_ipn[:i]*eig_ref[i]
        eig_t_ipn[i]  = 
        