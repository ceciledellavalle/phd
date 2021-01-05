"""
Classes and Functions used in the model.
Classes
-------
    MyMatmul : Multiplication with a kernel (for single or batch)
    Physics       : 

Functions
-------
    Sinkhorn_loss :

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
    Attributes
    ----------
        nx         (int): size of initial signal
        m          (int): size of eigenvectors span
        a          (int): oder of ill-posedness 
        p          (int): order of regularisation
        basis (np.array): transformation between signal and eigenvectors basis
    """
    def __init__(self,nx,m=200,a=1,p=1):
        # Physical parameters
        self.m  = m
        self.a  = a
        self.p  = p
        # Eigenvalues
        self.eigm= (np.linspace(0,m-1,m)+1/2)*np.pi
        # Basis transformation
        eig_m = self.eigm.reshape(-1,1)
        v1     = ((2*np.linspace(0,nx-1,nx)+1)/2/nx).reshape(1,-1)
        v2     = (np.ones(nx)/2/nx).reshape(1,-1)
        base   = np.zeros((m,nx))
        base   = 2*np.sqrt(2)*np.cos(v1*eig_m)*np.sin(v2*eig_m)/eig_m
        self.basis = base
        
    def BaseChange(self,x):
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
    
    def BaseChangeInv(self,x):
        """
        Change basis from eigenvectors span to signal.
        Parameters
        ----------
            x (np.array): signal of size nxcxm
        Returns
        -------
            (np.array): of size nxcxnx
        """
        return np.matmul(x,(self.basis))
    
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
       Top = np.diag(1/self.eigm**self.a)
       # D  = 2*np.diag(np.ones(nx)) - np.diag(np.ones(nx-1),-1) - np.diag(np.ones(nx-1),1)# matrice de dérivation
       Dop = np.diag(self.eigm**self.p)
       # Convert to o Tensor
       Trsf = torch.FloatTensor(Top)
       DtD = torch.FloatTensor(np.transpose(Dop).dot(Dop))
       TtT = torch.FloatTensor(np.transpose(Top).dot(Top))
       tensor_list = [DtD,TtT]
       return Trsf, tensor_list
      
    def Compute(self,x):
        """
        Compute the transformation by the Abek integral operator
        in the basis of eigenvectors.
        Parameters
        ----------
            x (np.array): signal of size nxcxm
        Returns
        -------
            (np.array): of size nxcxm
        """
        Top = np.diag(1/self.eigm**self.a)
        return np.matmul(x,Top)

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

#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
#
def Sinkhorn_loss(x, y, epsilon=0.01, niter=100):
    """
    Given two emprical measures defined on a uniform grid xi = yi = i/nl 
    (they are thue often refered to as "histograms"),
    outputs an approximation of the OT cost with regularization parameter epsilon
    niter is the max number of steps in sinkhorn loop
    Parameters
    ----------
        x (torch.FloatTensor): 1D-signal, size n*c*nx
        y (torch.FloatTensor): 1D-signal, size n*c*nx
    Returns
    -------
        (torch.FloatTensor): the Wasserstein distance, size n*c
    """

    # Definition of the cost matrix :
    _,_,nx = x.shape
    t = np.linspace(0,1,nx)
    [Y,X] = np.meshgrid(t,t)
    C_np = (X-Y)**2
    C = Variable(torch.FloatTensor(C_np), requires_grad=False)

    # The initial measures (histogram or marginal weigths)
    a = 1.*x
    b = Variable(1.*y, requires_grad=False)

    # Parameters of the Sinkhorn algorithm.
    tau = -.8  # nesterov-like acceleration
    thresh = 10**(-1)  # stopping criterion

    # Elementary operations 
    # .....................................................................
    def ave(u, u1):
        "Over-relaxation to accelerate the convergence of the fixed-point algorithm." 
        "It consists in replacing the update by a linear combination of the new and previous iterate. "
        return tau * u + (1 - tau) * u1

    def M(u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(1) + v.unsqueeze(0)) / epsilon

    def lse(A):
        "log-sum-exp"
        return torch.log(torch.exp(A).sum(1, keepdim=True) + 1e-6)  # add 10^-6 to prevent NaN

    # Actual Sinkhorn loop 
    # ......................................................................
    f, g = 0. * a, 0. * a
    actual_nits = 0 
    err = 0.

    for i in range(niter):
        f1 = f  # used to check the update error
        # Stable update u <- eps ( log a_i - log sum exp (-c_{ij} + f_i + g_j)/eps + f_i
        f = epsilon * (torch.log(a) - lse(M(f, g)).squeeze()) + f 
        # Stable update g <- eps ( log a_i - log sum exp (-c_{ij} + f_i + g_j)/eps + g_j
        g = epsilon * (torch.log(b) - lse(M(f,g).transpose(-2, -1)).squeeze()) + g
        # Error check
        err = (f - f1).abs().sum()
        actual_nits += 1
        if (err < thresh).data.numpy():
            break
            
    
    # Cost computatiom
    # ......................................................................
    F, G = f, g
    # Transport plan P_{ij} = a_i b_j exp (- C_{ij}+f_i+g_j )/\epsilon
    P = torch.exp(torch.log(a.unsqueeze(1))+ torch.log(b.unsqueeze(0)) + M(F, G))  
    # Sinkhorn cost
    cost = torch.sum(P * C)  

    return cost

