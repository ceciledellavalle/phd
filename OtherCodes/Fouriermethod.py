# general
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.linalg import inv,pinvh,eig,eigh
from scipy.special import gamma
import pandas as pd
from scipy.interpolate import interp1d
# local
from FBResNet.myfunc import Physics

class SolverChebyshev():
    def __init__(self,a=1):
        self.nx  = 2000
        self.m   = 50
        self.dx  = 1/(self.nx-1)
        # Matrice opérateur
        Ta = np.zeros((self.nx,self.nx))
        for i in range(self.nx):
            for j in range(self.nx):
                if j<=i:
                    Ta[i,j]=1/gamma(a)*1/a*self.dx**(a)*((i-j+1)**a-(i-j)**a)
        self.a  = a
        self.Ta = Ta
        
    def DataGen(self,idx,std_dev=0.05):
        # select the data
        folder = './Datasets/Signals'
        dfle = pd.read_csv(folder+'/'+'data_l_a1_cube.csv', sep=',',header=None)
        dfbn = pd.read_csv(folder+'/'+'data_bn_a1_cube_n{}'.format(std_dev)+'.csv', sep=',',header=None)
        # numpy array
        fx = np.array(dfle) # initial - elt basis
        fbn = np.array(dfbn)
        # transform the data
        x = fx[idx]
        y = fbn[idx]
        # adjoint
        Ty = np.transpose(self.Ta).dot(y)
        return x, y, Ty
        
    def Fourier_filter(self,idx,std_dev=0.05,cut=25,display=False):
        # Data generate
        x, y, Ty = self.DataGen(idx,std_dev=std_dev)
        # Fourier basis       
        h      = 1/(self.nx-1)
        eig    = (np.linspace(0,self.m-1,self.m)+1/2)*np.pi
        eig_m  = eig.reshape(-1,1)
        v1     = ((2*np.linspace(0,self.nx-1,self.nx)+1)*h/2).reshape(1,-1)
        v2     = (np.ones(self.nx)/2*h).reshape(1,-1)
        F      = 2*np.sqrt(2)/eig_m*np.cos(v1*eig_m)*np.sin(v2*eig_m)
        # Invertion
        yfft   = F.dot(Ty)
        xfft   = np.diag(eig**(2*self.a)).dot(yfft)
        xfft[cut:] = np.zeros(self.m-cut)
        coeff  = 1/gamma(self.a)**2
        x_pred = coeff*self.nx*np.transpose(F).dot(xfft)
        # plot
        if display:
            plt.plot(x_pred)
            plt.plot(x)
        # print info
        err = np.linalg.norm(x-x_pred)/np.linalg.norm(x)
        # retrun
        return err
        
    def AveragedError(self,std_dev=0.05):
        err = 0
        for idx in range(50):
            err += self.Fourier_filter(idx,std_dev=std_dev)
        return err/50
            