"""
Classes and Functions used in the model.
Classes
-------
    MyMatmul : Multiplication with a kernel (for single or batch)
    Physics  : 


@author: Cecile Della Valle
@date: 03/01/2021
"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
# General import
import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`

#

#
class Physics:
    """
    Define the physical parameters of the ill-posed problem.
    Alert : nx must be >> than m.
    Attributes
    ----------
        nx         (int): size of initial signal 
        m          (int): size of eigenvectors span
        a          (int): oder of ill-posedness 
        p          (int): order of regularisation
        basis (np.array): transformation between signal and eigenvectors basis
    """
    def __init__(self,nx,m=200,a=1,p=1):
        """
        Alert : nx must be >> than m.
        """
        # Physical parameters
        self.nx   = nx
        self.m    = m
        self.a    = a
        self.p    = p
        # Eigenvalues
        self.eigm = (np.linspace(0,m-1,m)+1/2)*np.pi
        # Basis transformation
        base       = np.zeros((self.m,self.nx))        
        h          = 1/(self.nx-1)
        eig_m      = self.eigm.reshape(-1,1)
        v1         = ((2*np.linspace(0,self.nx-1,self.nx)+1)*h/2).reshape(1,-1)
        v2         = (np.ones(self.nx)/2*h).reshape(1,-1)
        base       = 2*np.sqrt(2)/eig_m*np.cos(v1*eig_m)*np.sin(v2*eig_m)
        base[:,0]  = np.sqrt(2)/self.eigm*np.sin(h/2*self.eigm) # e_0
        base[:,-1] = 2*np.sqrt(2)/self.eigm*np.cos((1-h/4)*self.eigm)*np.sin(h/2*self.eigm)# e_nx
        self.basis = base
        # Operator T
        # step 0 : Abel operator integral
        # the image of the cos(t) basis is projected in a sin(t) basis
        Tdiag      = np.diag(1/self.eigm**self.a)
        # step 1 : From sin(t) basis to cos(t) basis
        eig_m      = self.eigm.reshape(-1,1)
        base_sin   = np.zeros((self.m,self.nx))
        base_sin   = 2*np.sqrt(2)/eig_m*np.sin(v1*eig_m)*np.sin(v2*eig_m)
        base_f     = np.matmul(self.basis,base_sin.T)
        # step 2 : Combinaison of Top and base change
        self.Top = np.matmul(base_f,Tdiag)*self.nx
        
    def BasisChange(self,x):
        """
        Change basis from signal to eigenvectors span.
        Parameters
        ----------
            x (np.array): signal of size nxcxnx
        Returns
        -------
            (np.array): of size nxcxm
        """
        return np.matmul(x,(self.basis).T)
    
    def BasisChangeInv(self,x):
        """
        Change basis from eigenvectors span to signal.
        Parameters
        ----------
            x (np.array): signal of size nxcxm
        Returns
        -------
            (np.array): of size nxcxnx
        """
        return np.matmul(x,self.basis*self.nx)
    
    def Operators(self):
       """
       Given a ill-posed problem of order a and a regularization of order p
       for a 1D signal of nx points,
       the fonction computes the tensor of the linear transformation Trsf
       and the tensor used in the algorithm.
       Returns
       -------
           (torch.FloatTensor): 
           (list)             :
    
       """
       # T  = 1/nx*np.tri(nx, nx, 0, dtype=int).T # matrice de convolution
       Top = np.diag(1/self.eigm**(self.a))
       # D  = 2*np.diag(np.ones(nx)) - np.diag(np.ones(nx-1),-1) - np.diag(np.ones(nx-1),1)# matrice de dérivation
       Dop = np.diag(self.eigm**(self.p))
       # Convert to o Tensor
       DtD = torch.FloatTensor(Dop*Dop)
       TtT = torch.FloatTensor(Top*Top)
       tensor_list = [DtD,TtT]
       return tensor_list
      
    def Compute(self,x):
        """
        Compute the transformation by the Abel integral operator
        in the basis of eigenvectors.
        Parameters
        ----------
            x (np.array): signal of size nxcxm
        Returns
        -------
            (np.array): of size nxcxm
        """
        return np.matmul(x,self.Top.T)
    
    def ComputeAdjoint(self,x):
        """
        Compute the transformation by the adjoint operator of Abel integral
        in the basis of eigenvectors.
        Parameters
        ----------
            x (np.array): signal of size nxcxm
        Returns
        -------
            (np.array): of size nxcxm
        """
        # We use the property of adjoint in discrete Hilbert space
        # < phi_n,T* phi_m > = < T phi_n, phi_m > 
        return np.matmul(x,self.Top)

#
class MyMatmul(nn.Module):
    """
    Performs 1D convolution with kernel
    Attributes
    ----------
        kernel (torch.FloatTensor): size nx*nx filter
    """
    def __init__(self, kernel):
        """
        Parameters
        ----------
            kernel (torch.FloatTensor): convolution filter
        """
        super(MyMatmul, self).__init__()
        self.kernel   = nn.Parameter(kernel.T,requires_grad=False)   
            
    def forward(self, x): 
        """
        Performs convolution.
        Parameters
        ----------
            x (torch.FloatTensor): 1D-signal, size n*c*nx
        Returns
        -------
            (torch.FloatTensor): result of the convolution, size n*c*nx
        """
        return torch.matmul(x.data,self.kernel)

