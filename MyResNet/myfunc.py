"""
iRestNet model classes.
Classes
-------
    MyMatmul : 

@author: Marie-Caroline Corbineau
@date: 03/10/2019
"""
import torch.nn as nn
import torch

class MyMatmul(nn.Module):
    """
    Performs 1D convolution with kernel
    Attributes
    ----------
        kernel (torch.FloatTensor): size nx*nx filter
        mode                 (str): 'single' or 'batch'
        stride               (int): dilation factor
        padding                   : instance of CircularPadding or torch.nn.ReplicationPad2d
    """
    def __init__(self, kernel):
        """
        Parameters
        ----------
            kernel (torch.FloatTensor): convolution filter
            mode                 (str): indicates if the input is a single image of a batch of images
        """
        super(MyMatmul, self).__init__()
        self.kernel   = nn.Parameter(kernel.T,requires_grad=False)   
            
    def forward(self, x): 
        """
        Performs convolution.
        Parameters
        ----------
            x (torch.FloatTensor): image(s), size n*c*nx
        Returns
        -------
            (torch.FloatTensor): result of the convolution, size n*c*h*w if mode='single', 
                                 size c*h*w if mode='batch'
        """
        return torch.matmul(x.data,self.kernel)
