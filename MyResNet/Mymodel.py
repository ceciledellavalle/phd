
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
from IPsolver import cadran
from modules import MyConv1d, TensorFilter, TransposeSquareFilter
from loss_ot import sinkhorn_loss


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
    
# Cnn_bar: to computing the barrier parameter
class Cnn_bar(nn.Module):
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
        super(Cnn_bar, self).__init__()
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


class IPIter(torch.nn.Module):
    """
    Computes the proximal interior point iteration.
    Attributes
    ----------
        DtD  (MyConv1d): 1-D convolution operators for the derivation
        TtT  (MyConv1d): 1-D convolution operator corresponding to TtT
        Mass (scalar):  maximal integral value
    """
    def __init__(self,im_range,a,p,dtype):
        """
        Parameters
        ----------
        Mass   (torch.FloatTensor): minimal and maximal pixel values
        a      (torch.FloatTensor): order of integration corresponding to Tt*T
        p      (torch.FloatTensor): convolution filter corresponding to derivation
        dtype                     : data type
        """
        super(IPIter, self).__init__()
        kernel = 0 #To compute
        D  = 0 #To compute
        self.DtD      = MyConv1d(D,'batch')
        self.TtT      = MyConv1d(kernel,'batch')
        self.Tt       = MyConv1d(kernel,'batch')
        self.im_range = im_range
        
    def Grad(self, reg, x, x_init):
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
        DtDx      = self.DtDv(x)
        Tt_x_init = self.Tt(x_init)
        TtTx      = self.TtT(x)
        return  TtTx - Tt_x_init + reg * DtDx

    def forward(self,gamma,mu,reg_mul,reg_constant,delta,x,x_init,std_approx,mode,save_gamma_mu_lambda):
        """
        Computes the proximal interior point iteration.
        Parameters
        ----------
      	    gamma                 (torch.nn.FloatTensor): stepsize, size 1
            mu                    (torch.nn.FloatTensor): barrier parameter, size n*1*1*1
            reg_mul, reg_constant (torch.nn.FloatTensor): parameters involved in the hidden layer used to estimate 
                                                          the regularization parameter, size 1
            delta                                (float): total variation smoothing parameter
            x                     (torch.nn.FloatTensor): images from previous iteration, size n*c*h*w
            Ht_x_blurred          (torch.nn.FloatTensor): Ht*degraded images, size n*c*h*w
            std_approx            (torch.nn.FloatTensor): approximation of the noise standard deviation, size n*1
            mode                                  (bool): True if training mode, False else
            save_gamma_mu_lambda                   (str): indicates if the user wants to save the values of the estimated hyperparameters, 
                                                          path to the folder to save the hyperparameters values or 'no' 
        Returns
        -------
       	    (torch.FloatTensor): next proximal interior point iterate, n*c*h*w
        """
        Dx         = torch.cat((self.Dv(x),self.Dh(x)),1)
        avg        = Dx.mean(-1).mean(-1).mean(-1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        reg        = reg_mul*std_approx/(torch.sqrt(((Dx-avg)**2).mean(-1).mean(-1).mean(-1))+reg_constant)
        x_tilde    = x - gamma*self.Grad(reg.unsqueeze(1).unsqueeze(2).unsqueeze(3), delta, x, Ht_x_blurred)
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
        return cardan.apply(gamma*mu,x_tilde,self.im_range,mode)
    

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
    def __init__(self,Mass,kernel_2,Dv,Dh,DvT,DhT,dtype):
        """
        Parameters
        ----------
        Mass                 (float): maximum of moment of order 1
        kernel_2 (torch.FloatTensor): convolution filter corresponding to Ht*H
        Dv       (torch.FloatTensor): convolution filter corresponding to the vertical gradient
        Dh       (torch.FloatTensor): convolution filter corresponding to the horizontal gradient
        DvT      (torch.FloatTensor): convolution filter corresponding to the transposed vertical gradient
        DhT      (torch.FloatTensor): convolution filter corresponding to the transposed horizontal gradient
        dtype                       : data type
        """
        super(Block, self).__init__()
        self.cnn_bar      = Cnn_bar()
        self.soft         = nn.Softplus()
        self.gamma        = nn.Parameter(torch.FloatTensor([1]))
        self.reg_mul      = nn.Parameter(torch.FloatTensor([-7])) 
        self.reg_constant = nn.Parameter(torch.FloatTensor([-5])) 
        self.IPIter       = IPIter(im_range,a,p,dtype)

    def forward(self,x,Ht_x_blurred,std_approx,save_gamma_mu_lambda):
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
        mu           = self.cnn_bar(x)  
        gamma        = self.soft(self.gamma)
        reg_mul      = self.soft(self.reg_mul)
        reg_constant = self.soft(self.reg_constant)
        return self.IPIter(gamma,mu,reg_mul,reg_constant,self.delta,x,Ht_x_blurred,std_approx,self.training,save_gamma_mu_lambda)
    
class myModel(torch.nn.Module):
    """
    iRestNet model.
    Attributes
    ----------
        Layers (torch.nn.ModuleList object): list of iRestNet layers
        Haar              (MyConv2d object): 2-D convolution operator computing Haar wavelet diagonal coefficients
    """
    def __init__(self,mass,a,p,nx,dtype,nL):
        super(myModel, self).__init__()
        self.Layers   = nn.ModuleList()
        Laplacian = 2*np.diag(np.ones(nx))-np.diag(np.ones(nx-1),-1)-np.diag(np.ones(nx-1),-1)
        D = 
        Ker =
        D, Ker  = TensorFilter([D, Kernel])
        for i in range(nL):
            self.Layers.append(Block(im_range,T,Tt,D,Dt,dtype))
        

    def forward(self,x,Ht_x_blurred,mode,block=0,std_approx=torch.tensor([-1]),save_gamma_mu_lambda='no'):
        """
        Computes the output of the layer.
        Parameters
        ----------
      	    x            (torch.nn.FloatTensor): previous iterate, size n*c*h*w
            Ht_x_blurred (torch.nn.FloatTensor): Ht*degraded image, size n*c*h*w
            std_approx   (torch.nn.FloatTensor): estimated noise standard deviation, size n*1 (default is torch.tensor([-1]))
            save_gamma_mu_lambda          (str): indicates if the user wants to save the values of the estimated hyperparameters, 
                                                 path to the folder to save the hyperparameters values or 'no'  (default is 'no')
        Returns
        -------
       	    (torch.FloatTensor): if mode='first_layer' or 'greedy' then it is the output of the current layer, 
                                 if mode='last_layers_lpp' or 'test' then it is the output of the network, size n*c*h*w
        """
        if (std_approx<0).any():
            # computes approximation of noise standard deviation
            # evaluation based on Section 11.3.1 of "S. Mallat. A wavelet tour of signal processing. Elsevier, 1999".
            y          = torch.abs(self.Haar(x)).view(x.data.shape[0],-1).data/0.6745
            std_approx = torch.topk(y,ceil(y.shape[1]/2),1)[0][:,-1]
        if mode=='first_layer' or mode=='greedy':
            x = self.Layers[block](x,Ht_x_blurred,std_approx,save_gamma_mu_lambda)
        elif mode=='test':
            for i in range(0,len(self.Layers)):
                # we use .detach() to avoid computing and storing the gradients since the model is being tested
                x = self.Layers[i](x.detach(),Ht_x_blurred,std_approx,save_gamma_mu_lambda) 
        return x
    

