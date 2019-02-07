# -*- coding : utf-8 -*-


######################### 1D TRANSPORT EQUATION USING FINITE DIFFERENCES  ###########################
#########################  EULER IMPLICIT UPSTREAM DOWNSTREAM SCHEME     ############################

### IMPORTATION PYTHON
import numpy as np
import math

def LaxWendroffScheme(L,NX,T,NT,speed,state_0):

    # Initialisation
    state_lw = np.zeros((NX,NT))
    state_lw[:,0] = state_0.copy()

    for k in range(0,NT-1):
        # CFL definition
        cfl = speed[k]*T/NT*NX/L
        # Computation of flow matix
        flow_lw =\
        (1-cfl**2)*np.eye(NX,NX)\
        - cfl/2*(1-cfl)*np.diag(np.ones(NX-1),1)\
        + cfl/2*(1+cfl)*np.diag(np.ones(NX-1),-1)
        # Computation of state
        state_lw[:,k+1] = np.dot(flow_lw,state_lw[:,k])

    return state_lw
