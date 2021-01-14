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
from FBResNet.myfunc import MyMatmul
from FBResNet.proxop.hypercube import cardan
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
    def __init__(self,exp):
        """
        Parameters
        ----------
        exp         (Physic object) : contains the experimental parameters   
        """
        super(Block, self).__init__()
        #
        self.nx       = exp.nx
        self.m        = exp.m
        # if m is big enough, we cut high frequencies that correspond to noise
        # else m is inferior to nx the projection in cos basis functions as a regularisation
        if exp.m>=exp.nx/100: 
            self.cnn_reg = Cnn_reg(exp)
        self.reg      = torch.FloatTensor([0.0])
        self.gamma    = nn.Parameter(torch.FloatTensor([0.5]))
        self.cnn_mu   = Cnn_bar(exp.nx)
        self.soft     = nn.Softplus(100)
        #
        liste_op      = exp.Operators()
        self.tDD      = MyMatmul(liste_op[0])
        self.tTT      = MyMatmul(liste_op[1])
        self.Peig     = MyMatmul(liste_op[2]) # eltTocos
        self.Pelt     = MyMatmul(liste_op[3]) # cosToelt

    def Grad(self,reg,x,x_b):
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
        tDDx      = self.tDD(x)
        tTTx      = self.tTT(x)
        return tTTx - x_b + reg * tDDx

    def forward(self,x,x_b):
        """
        Computes the next iterate, output of the layer.
        Parameters
        ----------
      	    x     (torch.nn.FloatTensor): previous iterate, size n*c*h*w
            x_b   (torch.nn.FloatTensor): size n*c*nx
            mode_training         (bool): True if training mode, False else
            save_theta             (str): indicates if the user wants to save the values of the lipschitz cste
                                                 path to the folder to save the hyperparameters values or 'no' 
        Returns
        -------
       	    (torch.FloatTensor): next iterate, output of the layer, n*c*h*w
        """
        # set parameters
        # Barrier parameter
        mu       = self.cnn_mu(self.Pelt(x)) 
        # Gradient descent parameter 
        gamma    = self.soft(self.gamma)
        # Regularisation parameter
        if self.m>=self.nx/100: 
            reg      = self.cnn_reg(x_b)
            self.reg = reg.clone().detach() # register for lipschitz cste
        else :
            reg      = self.reg
        # compute x_tilde
        x_tilde = x - gamma*self.Grad(reg, x, x_b)
        # project in finite element basis
        x_tilde = self.Pelt(x_tilde)
        # proximal operator
        x_tilde = cardan.apply(gamma*mu,x_tilde,self.training)
        # back to eigenvector cos basis
        x_tilde = self.Peig(x_tilde)
        return x_tilde 

    
class MyModel(torch.nn.Module):
    """
    iRestNet model.
    Attributes
    ----------
        Layers (torch.nn.ModuleList object): list of iRestNet layers
        nL                            (int): number of layers
        param               (Physic object): contains the experimental parameters
    """
    def __init__(self,exp,nL=50):
        super(MyModel, self).__init__()
        self.Layers   = nn.ModuleList()
        self.nL       = nL
        self.param    = exp
        #
        for _ in range(nL):
            self.Layers.append(Block(exp))
        

    def forward(self,x,x_b):
        """
        Computes the output of the layer.
        Parameters
        ----------
      	    x            (torch.nn.FloatTensor): previous iterate, size n*nx
            x_b          (torch.nn.FloatTensor): initial signal, size n*nx
        Returns
        -------
       	    (torch.FloatTensor): the output of the network, size n*c*h*w
        """
        for i in range(0,len(self.Layers)):
                x = self.Layers[i](x,x_b)
        return x
    
    # Lifschitz : Lifschitz constant of the network
    def Lifschitz(self,opt="semi"):
        """
        Given a ill-posed problem of order a and a regularization of order p
        for a 1D signal of nx points,
        the fonction Physics compute the tensor of the linear transformation Trsf
        and the tensor used in the algorithm.
        Parameters
        ----------
            model (MyModel): model of the neural network
            opt      (str) : "semi" to consider the semi-norm on the output
                             (or "total" to include the norm on the bias)
        Returns
        -------
            (float): Lifschitz constant theta of the neural network
    
        """
        # Step 0.0 : initialisation
        nL       = len(self.Layers) # number of layers
        m        = self.param.m  # dimension of the eigenvector space
        #
        eig_ref  = np.zeros((nL,m))
        eig_ip   = np.zeros((nL,m))
        eig_t_ip = np.zeros((nL,m))
        ai       = np.zeros(nL)
        theta    = 1.0
        # Step 1 : vectors of eigenvals of T^*T and D^*D
        eig_T = 1/self.param.eigm**(2*self.param.a)
        eig_D = self.param.eigm**(2*self.param.p)
        # Step 2 : on parcourt les layers du reseau
        with torch.no_grad():
            for i in range(nL-1,-1,-1):
                # Acces to the parameters of each layers
                gamma = self.Layers[i].gamma.detach().numpy()
                reg   = self.Layers[i].reg.detach().numpy()
                mu    = 1
                # Computes the ref eigenvals
                eig_ref[i,:] = 1 - gamma*(eig_T+reg*eig_D)
                # Step 2.0 Computes beta_i,p
                for p in range(0,m):
                    if i==nL-1:
                        eig_ip[i,p]   = eig_ref[i,p]
                        eig_t_ip[i,p] = gamma
                    else:
                        eig_ip[i,p]   = eig_ip[i+1,p]*eig_ref[i,p]
                        eig_t_ip[i,p] = eig_t_ip[i+1,p]+gamma*np.prod(eig_ref[i+1:,p])
                # Step 2.1 : compute ai
                if opt == "semi":
                     aip = eig_ip[i,:]**2 + eig_t_ip[i,:]**2
                if opt == "init":
                     aip = eig_ip[i,:]**2
                else :
                    aip  = eig_ip[i,:]**2 + eig_t_ip[i,:]**2 +1 \
                         + np.sqrt(( eig_ip[i,:]**2 + eig_t_ip[i,:]**2+1)**2 \
                         -  4* eig_ip[i,:]**2)
                ai[i] = 1/2*np.amax(aip)
                print("ai = ",np.sqrt(ai[i]))
            # Step 3 : compute theta
            print("calcul de theta")
            theta     = np.zeros(nL)
            theta[0]  = 1
            for i in range(0,nL):
                theta[i] = np.sum(theta*np.sqrt(ai))
                print(theta[i])
        # Step 3 : return
        return theta[-1]/(2**(nL-1))
        

# Cnn_reg: to compute the regularisation parameter
class Cnn_reg(nn.Module):
    """
    Predicts the regularisation parameter.
    Attributes
    ----------
        a    (torch.FloatTensor): 
        p    (torch.FloatTensor):
       cste  (torch.FloatTensor):
       m                   (int):
       nx                  (int):
        
    """
    def __init__(self,exp):
        super(Cnn_reg, self).__init__()
        self.a    = nn.Parameter(torch.FloatTensor([exp.a]),requires_grad=False)
        self.p    = nn.Parameter(torch.FloatTensor([exp.p]),requires_grad=False)
        self.Eig  = nn.Parameter(torch.FloatTensor(np.diag(exp.eigm)),requires_grad=False)
        #
        self.soft = nn.Softplus(100)
        self.inv  = MyMatmul(self.Eig**(2*self.a))
        #
        self.cste = nn.Parameter(torch.FloatTensor([0.001]),requires_grad=True)
        # numpy
        self.nx   = exp.nx
        self.m    = exp.m
         
    def forward(self,x_in):
        """
        Computes the barrier parameter.
        Parameters
        ----------
      	    x (torch.FloatTensor): images, size n*c*h*w 
        Returns
        -------
       	    (torch.FloatTensor): barrier parameter, size n*1*1*1
        """
        size             = x_in.size()
        x_out            = x_in.clone().detach() 
        x_out            = self.inv(x_out)
        x_fil            = x_out.clone().detach()
        nflt             = ceil(0.9/(1-1/100)*(self.m-self.nx/100))
        x_fil[:,:,nflt:] = torch.zeros((1,1,self.m-nflt))
        #
        delta            = torch.linalg.norm(x_out-x_fil)# estimation de l' erreur
        #
        cste             = self.soft(self.cste)
        x                = cste*delta**(2*(self.a+self.p)/(self.a+2))
        return x
    
# Cnn_bar: to compute the barrier parameter
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
    def __init__(self,nx):
        super(Cnn_bar, self).__init__()
        self.lin2   = nn.Linear(nx, 256)
        self.conv2  = nn.Conv1d(1, 1, 5,padding=2)
        self.conv3  = nn.Conv1d(1, 1, 5,padding=2)
        self.lin3   = nn.Linear(16*1, 1)
        self.avg    = nn.AvgPool1d(4, 4)
        self.soft   = nn.Softplus(100)

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