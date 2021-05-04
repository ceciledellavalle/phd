"""
SolverCheybyshev classe.

@author: Cecile Della Valle
@date: 03/01/2021
"""

# importation
import numpy as np
from torch.autograd import Variable
import torch
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.special import gamma

class SolverKalman():
    
    def __init__(self,a=1):
        self.nx  = 200
        self.dx  = 1/(self.nx-1)
        self.a   = a
        # Matrix Operator
        # Moment a = 1
        if self.a==1:
             self.op_1 = 1/self.nx*np.ones(self.nx)
        # Moment a = 1/2
        elif self.a==0.5:
             self.op_1 = 1/gamma(a)*1/self.nx*np.linspace(1/self.nx,1,self.nx)**(self.a-1)
        else:
            self.op1 = np.eye(nx)
        
    def Kalman_filter(self,y,alpha=0.1):
        """ Comnpute nx steps of Kalman filter 
        to reconstruct the initial condition :
        obs        -- real measurement (eventually noisy) (1 x nx)
        std_dev    -- standart deviation
        """
        #
        obs = np.flip(y)
        ### INITIALISATION
        # Construction of the observer (equal zero for the initial condition taken as parameter)
        obs_kalman = np.vstack((self.op_1, np.zeros(self.nx))).reshape(1,2*self.nx)


        # Construction of the new dynamic (identity for the initial condition)
        flow = np.diag(np.ones(self.nx-1),-1)
        flow_kalman = np.concatenate((\
                    np.concatenate((flow,np.zeros((self.nx,self.nx))),axis=1),\
                    np.concatenate((np.zeros((self.nx,self.nx)),np.eye(self.nx)),axis=1)))

        # Construction of the norm of the two spaces
        inv_norm_obs = alpha/self.nx*np.eye(1)


        # Initialisation of the covariance matrix
        cov_op_m = np.kron(np.ones((2,2)),np.eye(self.nx))
        cov_op_p = cov_op_m.copy()

        # Initialisation of the state
        state_m = np.zeros(2*self.nx)
        state_p = state_m.copy()
        state_kalman = np.zeros((2*self.nx,self.nx))

        ### KALMAN FILTER
        for k in range(0,self.nx):

            ### CORRECTION
            # Covariance computation +
            interim_matrix = inv_norm_obs + obs_kalman.dot(cov_op_m).dot(obs_kalman.transpose())
            kalman_gain = cov_op_m.dot(obs_kalman.transpose()).dot(np.linalg.inv(interim_matrix))

            cov_op_p = (np.eye(2*self.nx) - kalman_gain.dot(obs_kalman)).dot(cov_op_m)\
            .dot(np.transpose((np.eye(2*self.nx) - kalman_gain.dot(obs_kalman)))) \
            + kalman_gain.dot(inv_norm_obs).dot(np.transpose(kalman_gain))

            # State correction computation +
            state_p = state_m + kalman_gain\
            .dot(obs[k]- np.dot(obs_kalman,state_m))    

            ### PREDICTION
            # Covariance computation -
            cov_op_m = flow_kalman.dot(cov_op_p).dot(flow_kalman.transpose())

            # State prediction computation -
            state_m = np.dot(flow_kalman,state_p)
        
            # Saving the solution
            state_kalman[:,k] = state_m.copy()

        return state_kalman[self.nx:,-1]
        
    def DataGen(self,idx,std_dev=0.05):
        # select the data
        folder = './Datasets/Signals'
        dfle = pd.read_csv(folder+'/'+'data_l_a{}_cube.csv'.format(self.a), sep=',',header=None)
        dfbn = pd.read_csv(folder+'/'+'data_bn_a{}_cube_n{}'.format(self.a,std_dev)+'.csv', sep=',',header=None)
        # numpy array
        fx = np.array(dfle) # initial - elt basis
        fbn = np.array(dfbn)
        # transform the data
        x = fx[idx]
        y = fbn[idx]
        # interpolate
        t  = np.linspace(0,1-1/2000,2000)
        ti = np.linspace(0,1-1/self.nx,self.nx)
        fx = interp1d(t,x)
        xi = fx(ti)
        fy = interp1d(t,y)
        yi = fy(ti)
        return xi, yi
        
    def AveragedError(self,std_dev=0.05,cut=0.02):
        err = 0
        for idx in range(50):
            x, y = self.DataGen(idx,std_dev=std_dev)
            xp   = self.Kalman_filter(y,alpha=cut)
            err += np.linalg.norm(xp-x)/np.linalg.norm(x)
        return err/50
    
    def Test_kalman(self,noise = 0.1,cut = 0.002):
        # data
        x, y = self.DataGen(15,std_dev=noise)
        #
        xp = self.Kalman_filter(y,alpha=cut)
        # plot
        t = np.linspace(0,1,self.nx)
        plt.plot(t,y)
        plt.plot(t,x)
        plt.plot(t,xp)
        plt.show()
        print("x-xp/x =",np.linalg.norm(xp-x)/np.linalg.norm(x))