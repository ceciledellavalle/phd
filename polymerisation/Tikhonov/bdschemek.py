# -*- coding : utf-8 -*-


######################### 1D TRANSPORT EQUATION USING FINITE DIFFERENCES  ###########################
#########################  EULER IMPLICIT UPSTREAM DOWNSTREAM SCHEME     ############################

### IMPORTATION PYTHON
import numpy as np
import math


### FUNCTION DEFINITION
def BeckerDoringScheme(L,NX,T,NT,a,b,c0,state_init):

    ### INTERMEDIATE MATRIX INITIALISATION
    alpha = NX/L*T/NT # Pseudo CFL
    A = alpha*a*(np.eye(NX-1) - np.diag(np.ones(NX-2),-1))
    B = alpha*b*(np.eye(NX-1) - np.diag(np.ones(NX-2),1))


    ### INITIAL CONDITION
    x = np.linspace(L/NX,L,NX-1)
    # System state y
    state = np.zeros((NX,NT))
    state[1:,0] = state_init[:-1] # Gaussienne
    state[0,0] = c0

    # Computation of the constant total mass
    masse_rho = L/NX*np.linspace(L/NX, L, NX).dot(state[:,0])


    ### ITERATION PARAMETERS
    iter_max = 100
    s  = 0
    et = L/NX*np.linspace(2*L/NX, L, NX-1)
    ds = np.zeros(NX-1)

    ### WE COMPUTE DIRECT SOLUTION # BD IMPLICIT
    for n0 in range(0,NT-1):

        # Initialisation of increment delta
        delta = 100*np.ones(NX-1)
        s = 0
        # Set c(s=0) = cn0
        state[:,n0+1] = state[:,n0].copy()

        while (s<iter_max)&(np.amax(delta)>0.001):

            # Intermediate matrix dF = [1,et];[ds,Ts]
            Ts = np.eye(NX-1,NX-1) + state[0,n0+1]*A + B
            Tsinv = np.linalg.inv(Ts)
            ds[1:] = a*state[2:,n0+1]-a*state[1:-1,n0+1]
            ds[0] = a*state[1,n0+1] - 2*a*state[0,n0+1]
            ds = NX/L*T/NT*ds
            Fhat = -(state[1:,n0+1] - state[1:,n0] \
            + (state[0,n0+1]*A+B).dot(state[1:,n0+1]))
            Fhat[0] = Fhat[0] + alpha*a*state[0,n0+1]**2

            # Variation of the monomers
            delta1 = masse_rho - L/NX*np.linspace(L/NX, L, NX).dot(state[:,n0+1]) \
            - et.dot(Tsinv).dot(Fhat)
            delta1 = delta1/(L/NX-et.dot(Tsinv).dot(ds))
            # variations of the polymers
            delta = Tsinv.dot(Fhat) - delta1*Tsinv.dot(ds)

            # Computation of the solution
            state[0,n0+1] = state[0,n0+1] + delta1
            state[1:,n0+1] = state[1:,n0+1] + delta

            # Incrementing iteration step
            s=s+1

    state_bd = np.concatenate((state[1:,:],np.zeros((1,NT))),axis=0)

    return state_bd

##################################################################################
##################################################################################
# Compitation of Speed vector with BD Scheme output
def SpeedComputation(L,NX,T,NT,a,b,c0,state0):
    ### OBSERVER - MOMENT OPERATOR
    # First moment
    observer_1= L/NX*np.linspace(L/NX, L-L/NX, NX-1)

    ### MOMENT 1rst and 2nd order
    mu1 = np.dot(observer_1,state0[1:,:])

    ### SPEED Computation
    masse_rho = c0 + mu1[0]
    speed0 = a*(masse_rho-mu1)\
    -b*np.ones(NT)

    return speed0
