# -*- coding : utf-8 -*-


#########################     LUENBERGER FILTER - STATE RECONSTITION     ###########################
#########################         LIFSCHITZ - SLYOSOV EQUATION           ###########################

### IMPORTATION PYTHON
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import sys
import scipy.optimize as opt

### IMPORTATION OF MY FUNCTION
from bdscheme import BeckerDoringScheme
from plotdynamic import PlotDymamicSolution
from lscarctmethod import LifshitzSlyosovScheme
from lscarctmethod import LSonestep


### NUMERICAL PARAMETERS
NT = 200  # Number of time steps
NX = 100  # Initial number of grid points


### PHYSICAL PARAMETERS
a = 0.1*np.ones(NX) # Polymerisation speed
a[NX-1] = 0
b = 0.5*np.ones(NX)  # Depolymerisation speed
b[0] = 0
L = 10.0  # Domain size
T = 10.0  # Integration time
c0 = 1000 # Concentration of monomers in the system
cmax = 1.0# Maximum concentration of polymers
imax = 2*L/3

### WE COMPUTE BECKER DORING SOLUTION ################################################################
### EULER IMPLICIT UPSTREAM DOWNSTREAM SCHEME 

state, masse_rho = BeckerDoringScheme(L,NX,T,NT,a,b,c0,cmax,imax)

### WE PLOT DIRECT SOLUTION ##########################################################################

stateBDanim = PlotDymamicSolution(L,11,np.linspace(L/NX, L, NX),state,NT,\
np.linspace(0, T, NT))


###################################################################################################
###################################################################################################
###################################################################################################

###############################     LUENBERGER      ###############################################

### WE COMPUTE SPEED ############################################################################

### DEPOLYMERISATION POLYMERISATION SPEED (known)
et = L/NX*np.linspace(2*L/NX, L, NX-1)
speeddt = a[5]*(masse_rho*np.ones(NT) - et.dot(state[1:,:-1]))\
- b[5]*np.ones(NT)

fig0, ax0 = plt.subplots()
ax0.plot(np.linspace(0,T,NT), speeddt)
ax0.set(xlabel='time (s)', ylabel='speed',
       title='Speed estimation (simulation of data)')
ax0.grid()

### WE COMPUTE DIRECT SOLUTION # CARACTERISTIC METHOD ##############################################
### SPEED VARIATING WITH TIME
state_sim, speedv, vectT = LifshitzSlyosovScheme(L,NX,T,NT,state,speeddt)

### PLOT SIMULATED SOLUTION
stateSIManim = PlotDymamicSolution(L,11,np.linspace(L/NX, L, NX),state_sim,NT,vectT)


### OBSERVER - MOMENT OPERATOR ######################################################################

# First moment
operator_moment_1= L/NX*np.linspace(L/NX, L, NX)
# Second moment
operator_moment_2= L/NX*np.square(np.linspace(L/NX, L, NX))
# Concatenate
observer = np.vstack((operator_moment_1, operator_moment_2))
### COMPUTE 1rst and 2nd MOMENT 
# exclusion of x=0 because it is the monomers concentration
mu = observer[:,1:].dot(state_sim[1:,:]) 

### WE COMPUTE KSI ESTIMATOR #######################################################################
### PARAMETER DEFINITION
Ncvgs = 10 # number of converge eigenvalue tried
cvgs = np.linspace(-0.001,-1,Ncvgs) # neg eigen value - convergence speed of the Luenberger

### TIME STEP VECTOR
vectdtn = np.zeros(NT)
vectdtn[0] = vectT[0]
vectdtn[1:] = np.diff(vectT)

### LUENBERGER ESTOMATOR
# Estimator
ksi = np.zeros((NT,Ncvgs))
# Adjoint/kern defined by T = int kern(x,y)*u(x) dx
kern = np.zeros((NX-1,NT,Ncvgs))

### COMPUTATION OF THE ESTIMATOR 
for j in range(0,NT-1):
    deltat = vectdtn[j]
    # Dynamic computation
    ksi[j+1,:] = 1/(1-cvgs*deltat)*(ksi[j,:] + deltat*mu[0,j]\
    + deltat*mu[1,j])

### COMPUTE kern Solution of 1D transport equation (SPEED speeddt)
### EXPLICIT FINITE ELEMENT METHOD
for j in range(0,NT-1):
    # Dynamic computation (explicit downstream)
    # + lambda*kern_i (n)
    # + x_i ^2 = ((i+1) L/NX )^2 
    # + x_i = ((i+1) L/NX )


  ###################################################################################################
  ###################################################################################################
  ###################################################################################################
  ##########################     FUCKING DOESNT WORK !!!!!!
    ##### TEST
    NX=10
    print(np.kron(np.linspace(2*L/NX, L, NX-1),np.ones((1,Ncvgs))))

    if speedv[j]<0 :
        way= 1
    else :
        way  = -1

    kern[:,j+1,:] = np.diag(np.ones(NX-2),way).dot(kern[:,j,:])\
    + vectdtn[j]*kern[:,j,:].dot(cvgs)\
    + vectdtn[j]*np.kron(np.linspace(2*L/NX, L, NX-1),np.ones(Ncvgs))\
    + vectdtn[j]*np.kron(np.square(np.linspace(2*L/NX, L, NX-1)),np.ones(Ncvgs))

    
### COMPUTE Ty = sum a y (L/NX)
Tkerny = L/NX*np.transpose(kern).dot(state_sim[1:,:])
Tkerny = np.diag(Tkerny,0)

### CHECK THE CONVERGENCE ##########################################################################
### Ty - ksi tends to 0
### CHECK DIFFERENCE BETWEEN ksi and Tkerny
fig1, ax1 = plt.subplots()
ax1.plot(np.linspace(0,T,NT), ksi, 'k--', label = 'Model estimation')
ax1.plot(np.linspace(0,T,NT), Tkerny,'k:', label = 'Solution')
legend = ax1.legend(loc='upper center', shadow=True, fontsize='x-large')
legend.get_frame().set_facecolor('C0') # Put a nicer background color on the legend.
ax1.set(xlabel='time (s)', ylabel='convergence',
       title='Difference between solution and estimator')
ax1.grid()

plt.show()
sys.exit()

### SOLUTION RECONSTITUTION #######################################################################
### BFGS OPTIMIZATION METHOD
####### kern           (NX x NT) -> kern of the integrate function
####### ksi            (1 x NT) -> luenberger observer, image of the kern function
####### state_recst    (NX x NT) -> reconstructed state from the Tikhonov regulation
####### tknov            (NT)    -> Tikhonov parameter

def Rosen(j,y,f,L,NX,kerna,regul,speed):
    """Function to be minimized
    parameters :
    j -- indicator
    y --  (NXxNT) state
    f -- (NT) ksi, luenberger observer
    kerna -- (NXxNT) kern of the integretion function
    yapriori -- (NX) transport of the previous optimization result state_recst[:,n-1] 
    regul -- (NT) Tikhonov coefficient and regulation
    speed -- (NXxNT) vector of speed according to variable time step """

    # initialisation
    f1 = np.zeros(NT)
    f2 = np.zeros(NT)
    yapriori = np.zeros[NX,NT] 

    for j in range(0,NT):
        
        # Compute fonction for step j
        f1[j] = L/NX*np.transpose(kerna[:,j]).dot(y[:,j]) - f[j]
        f2[j] = np.transpose(y[:,j]-yapriori).dot(y[:,j]-yapriori)
        func = f1 + regul*f2

        # Compute yapriori for step j+1
        ### UTILE QUE SI TU UTILISES MINIMIZE !!!
        ### A REFLECHIR

        if speed[j] >0 :
            yapriori = np.diag(np.ones(NX-2),-1).dot(yapriori)
        else :
            yapriori = np.diag(np.ones(NX-2),1).dot(yapriori)

    return func

def Rosen_der(dy,y,f,L,NX,kerna,regul,speed):
    """Function to be minimized
    parameters :
    dy --  (NXxNT) state perturbation
    y --  (NXxNT) state
    f -- (NT) ksi, luenberger observer
    kerna -- (NXxNT) kern of the integretion function
    yapriori -- (NX) transport of the previous optimization result state_recst[:,n-1] 
    regul -- (NT) Tikhonov coefficient and regulation
    speed -- (NT) vector of speed according to variable time step """









#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################

######################        THE END

#################################################################################################

plt.show()
