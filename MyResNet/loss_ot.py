"""
Author : Cécile Della Valle
Discrete OT : Sinkhorn algorithm for histogram.
"""

import torch
import numpy as np
from torch.autograd import Variable


def sinkhorn_loss(x, y, epsilon=0.01, niter=100):
    """
    Given two emprical measures defined on a uniform grid xi = yi = i/nl 
    (they are thue often refered to as "histograms"),
    outputs an approximation of the OT cost with regularization parameter epsilon
    niter is the max. number of steps in sinkhorn loop
    """

    # Definition of the cost matrix :
    batch,c,nx = x.shape
    t = np.linspace(0,1,nl)
    [Y,X] = np.meshgrid(t,t)
    C_np = (X-Y)**2
    C = Variable(torch.from_numpy(C_np), requires_grad=False)

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

