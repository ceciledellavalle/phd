
"""
iRestNet model classes.
Classes
-------
    WASS_loss  : 
    Cnn_bar    : predicts the barrier parameter
    IPIter     : computes the proximal interior point iteration
    Block      : one layer in iRestNet
    myModel    : iRestNet model

@author: Marie-Caroline Corbineau
@date: 03/10/2019
"""
# General import
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from torch.nn.modules.loss import _Loss
from math import ceil
import os
# Local import
from proxsolver import cadran
from loss_ot import sinkhorn_loss
from myfunc import MyMatmul


class WASS_loss(_Loss):
    """
    Defines the Wasserstein training loss.
    Attributes
    ----------
        ssiwass (method): function computing the Wasserstein distance
    """
    def __init__(self): 
        super(WASS_loss, self).__init__()
        self.wass = sinkhorn_loss()
 
    def forward(self, input, target):
        """
        Computes the training loss.
        Parameters
        ----------
      	    input  (torch.FloatTensor): restored signal, size batch*c*nx
            target (torch.FloatTensor): ground-truth signal, size batch*c*nx
        Returns
        -------
       	    (torch.FloatTensor): SSIM loss, size 1 
        """
        return self.wass(input,target)
    


# Proximal step
class IPIter(torch.nn.Module):
    """
    Computes the proximal interior point iteration.
    Attributes
    ----------
        DtD  (MyConv1d): 1-D convolution operators for the derivation
        TtT  (MyConv1d): 1-D convolution operator corresponding to TtT
        Mass (scalar):  maximal integral value
    """
    def __init__(self,mass,tensor_list):
        """
        Parameters
        ----------
        mass   (torch.FloatTensor): minimal and maximal pixel values
        tensor_list         (list): list of two tensors DtD and TtT
        dtype                     : data type
        """
        super(IPIter, self).__init__()
        self.DtD      = tensor_list[0]
        self.TtT      = tensor_list[1]
        self.mass     = mass
        
    def Grad(self, reg, x, x_b):
        """
        Computes the gradient of the smooth term in the objective function (data fidelity + regularization).
        Parameters
        ----------
      	    reg        (torch.FloatTensor): regularization parameter, size batch*1*1*1
            x            (torch.nn.Tensor): images, size batch*c*nx
            x_init       (torch.nn.Tensor):result of Ht applied to the degraded images, size batch*c*nx
        Returns
        -------
       	    (torch.FloatTensor): gradient of the smooth term in the cost function, size batch*c*nx
        """
        DtDx      = MyMatmul(self.DtDv,x)
        TtTx      = MyMatmul(self.TtT,x)
        return  TtTx - x_b + reg * DtDx

    def forward(self,gamma,mu,reg_mul,reg_constant,x,x_b,mode,save_gamma_mu_lambda):
        """
        Computes the proximal interior point iteration.
        Parameters
        ----------
      	    gamma                 (torch.nn.FloatTensor): stepsize, size 1
            mu                    (torch.nn.FloatTensor): barrier parameter, size n*1*1*1
            reg                   (torch.nn.FloatTensor): parameters involved in the hidden layer used to estimate 
                                                          the regularization parameter, size 1
            x                     (torch.nn.FloatTensor): images from previous iteration, size n*c*h*w
            x_b          (torch.nn.FloatTensor): Ht*degraded images, size n*c*h*w
            mode                                  (bool): True if training mode, False else
            save_gamma_mu_lambda                   (str): indicates if the user wants to save the values of the estimated hyperparameters, 
                                                          path to the folder to save the hyperparameters values or 'no' 
        Returns
        -------
       	    (torch.FloatTensor): next proximal interior point iterate, n*c*h*w
        """
        reg        = 0 #function de reg constant et reg mul
        x_tilde    = x - gamma*self.Grad(reg, x, x_b)
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
        return cardan.apply(gamma*mu,x_tilde,self.mass,mode)
    
# One layer
# Parameters of the network
class Block(torch.nn.Module):
    """
    One layer in iRestNet.
    Attributes
    ----------
        cnn_bar                           (Cnn_bar): computes the barrier parameter
        soft                    (torch.nn.Softplus): Softplus activation function
        gamma                (torch.nn.FloatTensor): stepsize, size 1 
        reg_mul,reg_constant (torch.nn.FloatTensor): parameters for estimating the regularization parameter, size 1
        IPIter                             (IPIter): computes the next proximal interior point iterate
    """
    def __init__(self,mass,tensor_list):
        """
        Parameters
        ----------
        mass            (float): maximum of moment of order 1
        tensor_list     (list) : 
        """
        super(Block, self).__init__()
        self.cnn_mu    = Cnn_param()
        self.cnn_reg   = Cnn_param()
        self.soft      = nn.Softplus()
        self.gamma     = nn.Parameter(torch.FloatTensor([1]))
        self.IPIter       = IPIter(mass,tensor_list)

    def forward(self,x,x_b,std_approx,save_gamma_mu_lambda):
        """
        Computes the next iterate, output of the layer.
        Parameters
        ----------
      	    x            (torch.nn.FloatTensor): previous iterate, size n*c*h*w
            Ht_x_blurred (torch.nn.FloatTensor): Ht*degraded image, size n*c*h*w
            std_approx   (torch.nn.FloatTensor): approximate noise standard deviation, size n*1
            save_gamma_mu_lambda          (str): indicates if the user wants to save the values of the estimated hyperparameters, 
                                                 path to the folder to save the hyperparameters values or 'no' 
        Returns
        -------
       	    (torch.FloatTensor): next iterate, output of the layer, n*c*h*w
        """
        mu       = self.cnn_mu(x)  
        gamma    = self.soft(self.gamma)
        reg      = self.cnn_reg(x)
        return self.IPIter(gamma,mu,reg,x,x_b,self.training,save_gamma_mu_lambda)
    
class myModel(torch.nn.Module):
    """
    iRestNet model.
    Attributes
    ----------
        Layers (torch.nn.ModuleList object): list of iRestNet layers
        Haar              (MyConv2d object): 2-D convolution operator computing Haar wavelet diagonal coefficients
    """
    def __init__(self,mass,tensor_list,nL):
        super(myModel, self).__init__()
        self.Layers   = nn.ModuleList()
        self.nL       = nL
        #
        for i in range(nL):
            self.Layers.append(Block(mass,tensor_list))
        

    def forward(self,x,x_b,mode,save_gamma_mu_lambda='no'):
        """
        Computes the output of the layer.
        Parameters
        ----------
      	    x            (torch.nn.FloatTensor): previous iterate, size n*c*h*w
            x_b          (torch.nn.FloatTensor): initial signal, size n*c*h*w
            mode                          (str): 'train' or 'test'
            save_gamma_mu_lambda          (str): indicates if the user wants to save the values of the estimated hyperparameters, 
                                                 path to the folder to save the hyperparameters values or 'no'  (default is 'no')
        Returns
        -------
       	    (torch.FloatTensor): if mode='train' then it is the output of the current layer, 
                                 if mode='test' then it is the output of the network, size n*c*h*w
        """
        if mode=='train' :
            for i in range(0,len(self.Layers)):
                x = self.Layers[i](x,x_b,save_gamma_mu_lambda)
        elif mode=='test':
            for i in range(0,len(self.Layers)):
                # we use .detach() to avoid computing and storing the gradients since the model is being tested
                x = self.Layers[i](x.detach(),x_b,save_gamma_mu_lambda) 
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
    def __init__(self):
        super(Cnn_param, self).__init__()
        # in_channels: int,
        # out_channels: int, 
        # kernel_size: Union[T, Tuple[T]], 
        # stride: Union[T, Tuple[T]] = 1, 
        # padding: Union[T, Tuple[T]] = 0, 
        # dilation: Union[T, Tuple[T]] = 1,
        # groups: int = 1, 
        # bias: bool = True, padding_mode: str = 'zeros
        self.conv2  = nn.Conv1d(1, 1, 5,padding=2)
        self.conv3  = nn.Conv1d(1, 1, 5,padding=2)
        # What is the size of the output ?
        self.lin    = nn.Linear(16*1, 1)
        self.avg    = nn.AvgPool1d(4, 4)
        self.soft   = nn.Softplus()

    def forward(self, x):
        """
        Computes the barrier parameter.
        Parameters
        ----------
      	    x (torch.FloatTensor): images, size n*c*h*w 
        Returns
        -------
       	    (torch.FloatTensor): barrier parameter, size n*1*1*1
        """
        x = self.soft(self.avg(self.conv2(x)))
        x = self.soft(self.avg(self.conv3(x)))
        x = x.view(x.size(1), -1)
        x = self.soft(self.lin(x))
        x = x.view(x.size(0),-1,1,1)
        return x