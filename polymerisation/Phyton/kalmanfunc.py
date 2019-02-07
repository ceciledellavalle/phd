# -*- coding : utf-8 -*-

## KALMAN FILTER
## DEPOLYMERISATION ONLY

### IMPORTATION
import numpy as np
import math
import sys

###############################################################################
## KALMAN FILTER
## LAX-WENDROFF SCHEME
###############################################################################

def KalmanLW(L,NX,T,NT,mu,speed,state_0,flow_lw,observer,norm):
    ### Initialisation

    # Construction of the new dynamic (identity for the initial condition)
    flow_kalman = np.concatenate((\
    np.concatenate((flow_lw,np.zeros((NX,NX))),axis=1),\
    np.concatenate((np.zeros((NX,NX)),np.eye(NX)),axis=1)))

    # Construction of the observer (equal zero for the initial condition
    # taken as parameter)
    observer_kalman = np.vstack((observer[:NX], np.zeros(NX), \
    observer[NX+1:], np.zeros(NX)))\
    .reshape(2,2*(NX))

    # Initialisation of the covariance matrix
    covariance_operator_m = np.kron(np.ones((2,2)),np.eye(NX))
    covariance_operator_p = covariance_operator_m.copy()

    # Initialisation of the state
    x = np.linspace(0,L,NX)
    state_m = np.kron(np.ones(2),state_0)
    state_p = state_m.copy()
    state_kalman = np.zeros((2*NX,NT))

    ### KALMAN FILTER
    for k in range(0,NT):
        # Saving the solution
        state_kalman[:,k] = state_m.copy()

        ### CORRECTION
        state_p, covariance_operator_p = \
        KalmanCorrection(\
        NX,mu[:,k],observer_kalman,norm,\
        state_m,covariance_operator_m)

        ### PREDICTION
        # Actualisation of the dynamic
        cfl = speed[k]*T/NT*NX/L
        flow_kalman[0:NX,0:NX] = \
        (1-cfl**2)*np.eye(NX,NX)\
        - cfl/2*(1-cfl)*np.diag(np.ones(NX-1),1)\
        + cfl/2*(1+cfl)*np.diag(np.ones(NX-1),-1)

        state_m, covariance_operator_m = \
        KalmanPrediction(\
        flow_kalman,\
        state_p,covariance_operator_p)

    return state_kalman[NX:,:]

#############################################################################
### KALMAN FILTER
### ADAPTED TIME SCALE
#############################################################################

def KalmanAdaptedTimeScale(L,NX,T,NT,mu,speed,state_0,flow,observer,norm):
    ### Initialisation
    # Construction of the observer (equal zero for the initial condition
    # taken as parameter)
    observer_kalman = np.vstack((observer[:NX], np.zeros(NX), \
    observer[NX+1:], np.zeros(NX)))\
    .reshape(2,2*(NX))
    # interpolation index for observer
    index = 0

    # Construction of the new dynamic (identity for the initial condition)
    flow_kalman = np.concatenate((\
    np.concatenate((flow,np.zeros((NX,NX))),axis=1),\
    np.concatenate((np.zeros((NX,NX)),np.eye(NX)),axis=1)))

    # Initialisation of the covariance matrix
    covariance_operator_m = np.kron(np.ones((2,2)),np.eye(NX))
    covariance_operator_p = covariance_operator_m.copy()

    # Initialisation of the state
    x = np.linspace(0,L,NX)
    state_m = np.kron(np.ones(2),state_0)
    state_p = state_m.copy()
    state_kalman = np.zeros((2*NX,NT))

    ### KALMAN FILTER
    for k in range(0,NT):
        # Saving the solution
        state_kalman[:,k] = state_m.copy()

        ### CORRECTION
        # refresh inv_norm_observation
        # M_n = \delta t_n M
        inv_norm_obs_dt = T/NT*abs(L/NX/speed[k])*norm[3]
        # interpolate muk
        mu_interp = mu[:,index]
        index += math.ceil(abs(L/NX/speed[k]))
        state_p, covariance_operator_p = \
        KalmanCorrection(\
        NX,mu_interp,observer_kalman,inv_norm_obs_dt,\
        state_m,covariance_operator_m)

        ### PREDICTION
        state_m, covariance_operator_m = \
        KalmanPrediction(\
        flow_kalman,\
        state_p,covariance_operator_p)

        return state_kalman[NX:,:]

###############################################################################
## STEPS OF THE KALMAN FILTER
###############################################################################

### CORRECTION
### ONE STEP KALMAN FILTER
def KalmanCorrection(\
NX,muk,obs_k,norm,\
state_m,covop_m):

    # Covariance computation +
    interim_matrix = norm \
    + obs_k.dot(covop_m)\
    .dot(obs_k.transpose())
    #
    gain_k = covop_m\
    .dot(obs_k.transpose())\
    .dot(np.linalg.inv(interim_matrix))
    #
    covop_p = (np.eye(2*NX) - gain_k.dot(obs_k))\
    .dot(covop_m)\
    .dot(np.transpose((np.eye(2*NX) - gain_k.dot(obs_k)))) \
    + gain_k.dot(norm).dot(np.transpose(gain_k))

    # State correction computation +
    state_p = state_m + gain_k\
    .dot(muk-np.dot(obs_k,state_m))

    return state_p, covop_p

### PREDICTION
### ONE STEP KALMAN FILTER
def KalmanPrediction(\
flow_k,\
state_p,covop_p):

    # Covariance computation -
    covop_m = flow_k.dot(covop_p)\
    .dot(flow_k.transpose())

    # State prediction computation -
    state_m = np.dot(flow_k,state_p)

    return state_m, covop_m

################################################################################
################################################################################
