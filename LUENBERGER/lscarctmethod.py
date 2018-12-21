# -*- coding : utf-8 -*-

######################### 1D TRANSPORT EQUATION USING CARACTERISTIC METHOD  #########################
#########################  EULER EXPLICIT UPSTREAM DOWNSTREAM SCHEME      ############################

### IMPORTATION PYTHON
import numpy as np
import math

### FUNCTION DEFINITION
def LifshitzSlyosovScheme(L,NX,T,NT,state,Gt) :

    ### INITIALIZATION
    # state_sim[0,:] stands for the concentration of c (check for coherence purpose)
    state_sim = np.zeros((NX,NT))
    state_sim[1:,0] = state[1:,0].copy() # Gaussienne
    # Time vector
    vectT = np.zeros(NT) # time step is not constant anymore
    idx = 0  # index walk on vecteur speed Gt
    speedv = np.zeros(NT) # transport speed is not constant anymore
    speedv[0] = Gt[0]

    for j in range(0,NT-1):

        deltat, state_sim[1:,j+1] = \
        LSonestep(L,NX,T,NT,Gt,idx,state_sim[1:,j])

        ### Computation of time t_n = delta t (n) + t_n-1
        vectT[j+1] = vectT[j] + deltat

        ### Computation of the new speed - interpolation on grid with step T/NT
        idx = (np.abs(np.linspace(0,T,NT)-vectT[j+1])).argmin()
        speedv[j+1] = Gt[idx]

    
        #if vectT[j+1] > T :
        #    print("that's all folks !")
        #    sys.exit()


    return state_sim, speedv, vectT


def LSonestep(L,NX,T,NT,speed,idx0,state_simj):
    ### Computation of corresponding delta t
    deltax = L/NX # space step
    dt = deltax/np.abs(speed[idx0])

    ### Translation of state with speed Gt[idx]
    if speed[idx0] >0 :
        state_simj1 = np.diag(np.ones(NX-2),-1).dot(state_simj)
    else :
        state_simj1 = np.diag(np.ones(NX-2),1).dot(state_simj)
        
    return dt, state_simj1