from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import sys

class cardan(torch.autograd.Function):  

    @staticmethod
    def forward(ctx,gamma_mu,xtilde,mass,u,device="cpu",mode_training=True):
        """
	    Finds the solution of the cubic equation involved in the computation of the proximity operator of the 
        logarithmic barrier of the hyperslab constraints (xmin< u^Tx <xmax) using the Cardano formula: x^3+ax^2+bx+c=0 
        is rewritten as x^3+px+q=0. Selects the solution x such that x-a/3 is real and belongs to ]xmin,xmax[.
        Parameters
        ----------
           gamma_mu (torch.FloatTensor): product of the barrier parameter and the stepsize, size n
           xtilde (torch.FloatTensor): point at which the proximity operator is applied, size n
           u_ker (torch.FloatTensor) : kernel conv u^T x_tilde, size n
           im_range (list): minimal and maximal pixel values
           device (string) : 
           mode_training (bool): indicates if the model is in training (True) or testing (False) (default is True)
        Returns
        -------
           sol (torch.FloatTensor): proximity operator of gamma_mu*barrier at xtilde, size n 
        """
        # Device CPU/GPU
        if device == "cuda":
            dtype = torch.cuda.FloatTensor
        else :
            dtype = torch.FloatTensor
        #initialize variables
        size              = xtilde.size()
        x1,x2,x3          = torch.zeros(1).type(dtype),torch.zeros(1).type(dtype),torch.zeros(1).type(dtype)   
        crit,crit_compare = torch.zeros(1).type(dtype),torch.zeros(1).type(dtype)
        sol               = torch.zeros(size).type(dtype),
        xmin              = 0
        xmax              = mass
        uTx               = torch.matmul(xtilde,u)
        #set coefficients
        a     = -(xmin+xmax+uTx)
        b     = xmin*xmax + uTx*(xmin+xmax) - 2*gamma_mu*torch.norm(u)**2
        c     = gamma_mu*(xmin+xmax) - uTx*xmin*xmax
        p     = b - (a**2)/3
        q     = c - a*b/3 + 2*(a**3)/27
        delta = (p/3)**3 + (q/2)**2  

        #three cases depending on the sign of delta
        #########################################################################
        #when delta is positive
        if delta>0:
            z1 = -q/2
            z2 = torch.sqrt(delta)
            w  = (z1+z2).sign() * torch.pow((z1+z2).abs(),1/3)
            v  = (z1-z2).sign() * torch.pow((z1-z2).abs(),1/3) 
            x1 = w + v   
            x2 = -(w + v)/2 ; #real part of the complex solution
            x3 = -(w + v)/2 ; #real part of the complex solution
        #########################################################################
        #when delta is 0
        elif delta==0:
            x1 = 3 *q / p 
            x2 = -1.5 * q / p
            x3 = -1.5 * q / p 
        #########################################################################
        #when delta is negative
        elif delta<0:
            cos = (-q/2) * ((27 / torch.pow(p,3)).abs()).sqrt() 
            cos[cos<-1] = 0*cos[cos<-1]-1
            cos[cos>1]  = 0*cos[cos>1]+1
            phi         = torch.acos(cos)
            tau         = 2 * ((p/3).abs()).sqrt() 
            x1     = tau * torch.cos(phi/3) 
            x2     = -tau * torch.cos((phi + np.pi)/3)
            x3     = -tau * torch.cos((phi - np.pi)/3)
        #########################################################################
        x1   = x1-a/3
        x2   = x2-a/3
        x3   = x3-a/3
        # when gamma_mu is very small there might be some numerical instabilities
        # in case there are nan values, we set the corresponding pixels equal to 2*xmax
        # these values will be replaced by valid values at least once
        if (x1!=x1).any():
            x1[x1!=x1]=2*xmax
        if (x2!=x2).any():
            x2[x2!=x2]=2*xmax
        if (x3!=x3).any():
            x3[x3!=x3]=2*xmax
        sol  = xtilde + (x1 - uTx)/torch.norm(u)**2*u
        #########################################################################
        #take x1
        p1 = sol
        uTp1 = torch.matmul(p1,u)
        if (uTp1>xmin)&(uTp1<xmax):
            crit[0] = -(torch.log(uTp1-xmin)+torch.log(xmax-uTp1))
            crit = 0.5*torch.norm(p1-xtilde)**2+gamma_mu*crit
        else:
            crit[0] = np.inf
        #########################################################################
        #test x2
        p2 = xtilde + (x2 - uTx)/torch.norm(u)**2*u
        uTp2 = torch.matmul(p2,u)
        if (uTp2 >xmin)&(uTp2 <xmax): 
            crit_compare[0]  = -(torch.log(uTp2-xmin)+torch.log(xmax-uTp2))
            crit_compare  = 0.5*torch.norm(p2-xtilde)**2+gamma_mu*crit_compare
        else:
            crit_compare[0] = np.inf
        if crit_compare<=crit:
            sol  = p2
            crit = crit_compare
        #########################################################################
        #test x3
        p3 = xtilde + (x3 - uTx)/torch.norm(u)**2*u
        uTp3 = torch.matmul(p3,u)
        if (uTp3>xmin)&(uTp3<xmax):
            crit_compare[0] = -(torch.log(uTp3-xmin)+torch.log(xmax-uTp3))
            crit_compare = 0.5*torch.norm(p3-xtilde)**2+gamma_mu*crit_compare
        else:
            crit_compare[0] = np.inf
        if crit_compare<=crit:
            sol  = p3
            crit = crit_compare
        #########################################################################
        # when gamma_mu is very small and xtilde is very close to one of the bounds,
        # the solution of the cubic equation is not very well estimated -> test xtilde
        # denom = (sol-xmin)*(sol-xmax)-2*gamma_mu -(sol-xtilde)*(xmin+xmax-2*sol)
        if (uTx>xmin)&(uTx<xmax):
            crit_compare = -(torch.log(xmax-uTx)+torch.log(uTx-xmin))
            crit_compare = gamma_mu*crit_compare
        else:
            crit_compare[0] = np.inf
        if crit_compare<crit :
            sol = xtilde
        
        if mode_training==True:
            ctx.save_for_backward(gamma_mu,xtilde,sol)
        return sol

    @staticmethod
    def backward(ctx, grad_output_var):
        """
        Computes the first derivatives of the proximity operator of the log barrier with respect to x and gamma_mu.
            This method is automatically called by the backward method of the loss function.
        Parameters
        ----------
           ctx (list): list of torch.FloatTensors, variable saved during the forward operation
           grad_output_var (torch.FloatTensor): gradient of the loss wrt the output of cardan
        Returns
        -------
           grad_input_gamma_mu (torch.FloatTensor): gradient of the prox wrt gamma_m 
           grad_input_u (torch.FloatTensor): gradient of the prox wrt x
           None: no gradient wrt the image range
           None: no gradient wrt the mode
        """
        xmin           = 0
        xmax           = 1
        dtype          = torch.cuda.FloatTensor
        grad_output    = grad_output_var.data
        gamma_mu,u,x   = ctx.saved_tensors
        denom          = (x-xmin)*(x-xmax)-2*gamma_mu -(x-u)*(xmin+xmax-2*x)
        
        idx                 = denom.abs()>1e-7
        denom[1-idx]        = denom[1-idx]+1
        grad_input_gamma_mu = (2*x-(xmin+xmax))/denom
        grad_input_u        = ((x**2-x*(xmin+xmax)+xmin*xmax))/denom
        # if denom is very small, it means that gamma_mu is very small and u is very close to one of the bounds,
        # there is a discontinuity when gamma_mu tends to zero, if 0<u<1 the derivative wrt x is approximately equal to 
        # 1 and the derivative wrt gamma_mu is approximated by 10^5 times the sign of 2*x[1-idx]-(xmin+xmax)
        grad_input_gamma_mu[1-idx] = 0*grad_input_gamma_mu[1-idx]+1e5*torch.sign(2*x[1-idx]-(xmin+xmax))
        grad_input_u[1-idx]        = 0*grad_input_u[1-idx]+1
        
        grad_input_gamma_mu = (grad_input_gamma_mu*grad_output).sum(1).sum(1).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        grad_input_u        = grad_input_u*grad_output
        
        # safety check for numerical instabilities
        if (grad_input_gamma_mu!=grad_input_gamma_mu).any():
            print('there is a nan in grad_input_gamma_mu')
            if (x!=x).any():
                print('there is a nan in x')
            sys.exit()
        if (grad_input_u!=grad_input_u).any():
            print('there is a nan in grad_input_u')
            sys.exit()
        
        grad_input_gamma_mu = Variable(grad_input_gamma_mu.type(dtype),requires_grad=True)
        grad_input_u        = Variable(grad_input_u.type(dtype),requires_grad=True)
        
        return grad_input_gamma_mu, grad_input_u, None, None