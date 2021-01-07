"""
RestNet model classes.
Classes
-------
    Block      : one layer in iRestNet
    myModel    : iRestNet model
    Cnn_bar    : predicts the barrier parameter

@author: Cecile Della Valle
@date: 03/01/2021
"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
# General import
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from math import ceil
import os
# Local import
from MyResNet.myfunc import MyMatmul
from MyResNet.proxop.hypercube import cardan
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# One layer
# Parameters of the network
class Block(torch.nn.Module):
    """
    One layer in myRestNet.
    Attributes
    ----------
        cnn_bar                           (Cnn_bar): computes the barrier parameter
        soft                    (torch.nn.Softplus): Softplus activation function
        gamma                (torch.nn.FloatTensor): stepsize, size 1 
        reg_mul,reg_constant (torch.nn.FloatTensor): parameters for estimating the regularization parameter, size 1
        DtD  (MyConv1d): 1-D convolution operators for the derivation
        TtT  (MyConv1d): 1-D convolution operator corresponding to TtT
        mass (scalar):  maximal integral value
    """
    def __init__(self,simu):
        """
        Parameters
        ----------
        simu     
        Warning ---> do not converge with bad initialization
        """
        super(Block, self).__init__()
        nx            = simu.nx
        self.cnn_mu   = Cnn_param(nx)
        self.reg      = torch.FloatTensor([0.0]) #nn.Parameter(torch.FloatTensor([0.0]))
        self.soft     = nn.Softplus()
        self.gamma    = nn.Parameter(torch.FloatTensor([1.0]))
        #
        liste_op      = simu.Operators()
        self.tDD      = MyMatmul(liste_op[0])
        self.tTT      = MyMatmul(liste_op[1])
        self.Peig     = MyMatmul(liste_op[2])
        self.Pelt     = MyMatmul(liste_op[3])

    def Grad(self, reg, x, x_b):
        """
        Computes the gradient of the smooth term in the objective function (data fidelity + regularization).
        Parameters
        ----------
      	    reg        (torch.FloatTensor): regularization parameter, size batch*1
            x            (torch.nn.Tensor): images, size batch*c*nx
            x_b          (torch.nn.Tensor):result of Ht applied to the degraded images, size batch*c*nx
        Returns
        -------
       	    (torch.FloatTensor): gradient of the smooth term in the cost function, size batch*c*nx
        """
        DtDx      = self.tDD(x)
        TtTx      = self.tTT(x)
        return  TtTx - x_b + reg * DtDx

    def forward(self,x,x_b,save_gamma_mu_lambda='no'):
        """
        Computes the next iterate, output of the layer.
        Parameters
        ----------
      	    x            (torch.nn.FloatTensor): previous iterate, size n*c*h*w
            x_b          (torch.nn.FloatTensor): size n*c*nx
            mode_training                (bool): True if training mode, False else
            save_gamma_mu_lambda          (str): indicates if the user wants to save the values of the estimated hyperparameters, 
                                                 path to the folder to save the hyperparameters values or 'no' 
        Returns
        -------
       	    (torch.FloatTensor): next iterate, output of the layer, n*c*h*w
        """
        # set parameters
        mu       = self.cnn_mu(self.Peig(x))  
        gamma    = self.soft(self.gamma)
        reg      = self.soft(self.reg)
        #save parameters
        if save_gamma_mu_lambda!='no':
            #write the value of the stepsize in a file
            file = open(os.path.join(save_gamma_mu_lambda,'gamma.txt'), "a")
            file.write('\n'+'%.3e'%gamma.data.cpu())
            file.close()
            #write the value of the barrier parameter in a file
            file = open(os.path.join(save_gamma_mu_lambda,'mu.txt'), "a")
            file.write('\n'+'%.3e'%mu.data.cpu())
            file.close()
            #write the value of the regularization parameter in a file
            file = open(os.path.join(save_gamma_mu_lambda,'lambda.txt'), "a")
            file.write('\n'+'%.3e'%reg.data.cpu())
            file.close()
        #
        # compute x_tilde
        x_tilde  = x - gamma*self.Grad(reg, x, x_b)
        # project in finite element basis
        x_tilde  = self.Peig(x_tilde)
        # proximal operator
        x        = cardan.apply(gamma*mu,x_tilde,self.training)
        # back to eigenvector cos basis
        return self.Pelt(x)

    
class MyModel(torch.nn.Module):
    """
    iRestNet model.
    Attributes
    ----------
        Layers (torch.nn.ModuleList object): list of iRestNet layers
        nL                            (int): number of layers
    """
    def __init__(self,simu,nL=50):
        super(MyModel, self).__init__()
        self.Layers   = nn.ModuleList()
        self.nL       = nL
        #
        for _ in range(nL):
            self.Layers.append(Block(simu))
        

    def forward(self,x,x_b,save_gamma_mu_lambda='no'):
        """
        Computes the output of the layer.
        Parameters
        ----------
      	    x            (torch.nn.FloatTensor): previous iterate, size n*nx
            x_b          (torch.nn.FloatTensor): initial signal, size n*nx
            save_gamma_mu_lambda          (str): indicates if the user wants to save the values of the estimated hyperparameters, 
                                                 path to the folder to save the hyperparameters values or 'no'  (default is 'no')
        Returns
        -------
       	    (torch.FloatTensor): the output of the network, size n*c*h*w
        """
        for i in range(0,len(self.Layers)):
                x = self.Layers[i](x,x_b,save_gamma_mu_lambda)
        return x
    

# Cnn_bar: to computing the barrier parameter
class Cnn_param(nn.Module):
    """
    Predicts the barrier parameter.
    Attributes
    ----------
        conv2, conv3 (torch.nn.Conv1d): 1-D convolution layer
        lin          (torch.nn.Linear): fully connected layer
        avg       (torch.nn.AVgPool2d): average layer
        soft       (torch.nn.Softplus): Softplus activation function
    """
    def __init__(self,nx):
        super(Cnn_param, self).__init__()
        self.lin2   = nn.Linear(nx, 256)
        self.conv2  = nn.Conv1d(1, 1, 5,padding=2)
        self.conv3  = nn.Conv1d(1, 1, 5,padding=2)
        self.lin3   = nn.Linear(16*1, 1)
        self.avg    = nn.AvgPool1d(4, 4)
        self.soft   = nn.Softplus()

    def forward(self, x_in):
        """
        Computes the barrier parameter.
        Parameters
        ----------
      	    x (torch.FloatTensor): images, size n*c*h*w 
        Returns
        -------
       	    (torch.FloatTensor): barrier parameter, size n*1*1*1
        """
        x = self.lin2(x_in)
        x = self.soft(self.avg(self.conv2(x)))
        x = self.soft(self.avg(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.soft(self.lin3(x))
        x = x.view(x.size(0),1,1)
        return x