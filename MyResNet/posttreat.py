"""
RestNet model classes.
Classes
-------
    

@author: Cecile Della Valle
@date: 03/01/2021
"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
# General import
import torch
import torch.nn as nn
import numpy as np
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`

    # Lifschitz : Lifschitz constant of the network
    def Lifschitz(self):
        """
        Given a ill-posed problem of order a and a regularization of order p
        for a 1D signal of nx points,
        the fonction Physics compute the tensor of the linear transformation Trsf
        and the tensor used in the algorithm.
        Parameters
        ----------
            model (MyModel): model of the neural network
        Returns
        -------
            (float): Lifschitz constant theta of the neural network
    
        """
        # Step 0.0 : initialisation
        nL       = len(self.Layers) # number of layers
        m        = self.m  # dimension of the eigenvector space
        #
        eig_ref  = np.zeros((nL,m))
        eig_ip   = np.zeros((nL,m))
        eig_t_ip = np.zeros((nL,m))
        ai       = np.zeros(nL)
        theta    = 1.0
        # Step 1 : vectors of eigenvals of T^*T and D^*D
        eig_T = (np.linspace(0,nx-1,nx)-1/2)*np.pi
        eig_D = 1/eig_T**p
        # Step 2 : on parcourt les layers du reseau
        for i in range(nL-1,-1,-1):
            # Acces to the parameters of each layers
            gamma = model.Layers[i].gamma.numpy()
            reg   = model.Layers[i].reg.numpy()
            mu    = 1
            # Computes the ref eigenvals
            eig_ref[i,:]    = 1 - gamma*(eig_T+reg*eig_D)
            # Step 2.0 Computes beta_i,p
            for p in range(0,m):
                if i==nL-1:
                    eig_ip[:i,p]   = eig_ref[i,p]
                    eig_t_ip[i,p]  = gamma
                else:
                    eig_ip[:i,p]   = eig_ip[i+1,p]*eig_ref[i,p]
                    eig_t_ip[i,p]  = eig_t_ip[i+1,p]+gamma*np.prod(eig_ref[i+1:,p])
            # Step 2.1 : compute ai
            aip = eig_ip[i,:]**2 + eig_t_ip[i,:]**2 +1 \
                    + np.sqrt(( eig_ip[i,:]**2 + eig_t_ip[i,:]**2+1)**2 \
                    -  4* eig_ip[i,:]**2)
            ai[i] = 1/2*np.amax(aip)
            # Step 2.2 : compute theta
            if i<0 :
                theta = theta*np.sqrt(ai[i])+1
            else :
                theta *= np.sqrt(ai[i])
        # Step 3 : return
        return theta/2**(nL-1)   
        