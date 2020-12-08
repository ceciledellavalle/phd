"""
Author : Cécile Della Valle
Discrete OT : Sinkhorn algorithm for histogram.
"""

import torch
import numpy as np
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss
#local import
from MyResNet.myfunc import Sinkhorn_loss

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


