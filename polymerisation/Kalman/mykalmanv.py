# -*- coding : utf-8 -*-

## KALMAN FILTER
## DEPOLYMERISATION ONLY

###################### 1D TRANSPORT EQUATION USING FINITE DIFFERENCES ##########################

### IMPORTATION
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import sys
##
from bdschemek import BeckerDoringScheme
from bdschemek import SpeedComputation
from laxwendroff import LaxWendroffScheme
from kalmanfunc import KalmanAdaptedTimeScale
from kalmanfunc import KalmanLW
from plotdynamic import PlotDymamicSolution

### PHYSICAL PARAMETERS
b = 5.0  # Depolymerisation speed
a = 1
c0 = 0
L = 3.0  # Domain size
T = 0.60  # Integration time

### NUMERICAL PARAMETERS
NT = 100  # Number of time steps
NX = 100   # Initial number of grid points

### BECKER DORING ###########################################################

### INITIAL CONDITION
x = np.linspace(0.0,L,NX)
cmax = 0.1 # Maximum concentration of polymers
imax = 1*L/2
sigma = L/20
def Gaussienne(x,c,i,s):
    gaussienne = c*np.exp(-(x-i)**2/(2*s**2)) # Gaussienne
    return gaussienne


### VECTOR INITIALISATION
# Initial condition set as a gaussian
state_init = Gaussienne(x,cmax,imax,sigma)

### SOLUTION OF BDSCHEME
state_bd = BeckerDoringScheme(L,NX,T,NT,a,b,c0,state_init)



speed = SpeedComputation(L,NX,T,NT,a,b,c0,state_bd)

### PLOTs
# Create plots with pre-defined labels.
fig0, ax0 = plt.subplots()
ax0.plot(np.linspace(0,T,NT), T/NT*NX/L*speed, 'r--', label='speed')
legend = ax0.legend(loc='lower right', shadow=True, fontsize='x-large')
ax0.set(xlabel='time (s)', ylabel='cfl',
       title='Becker Döring - total speed equivalency')
### LAX WENDROFF ###########################################################

state_lw = LaxWendroffScheme(L,NX,T,NT,speed,state_init)

### PLOTs
anim_lw = PlotDymamicSolution(1.1*L,1.1*cmax,\
np.linspace(0,L,NX),state_lw,
NT,np.linspace(0,T,NT))


### DATA GENERATION ###########################################################
### OBSERVER - MOMENT OPERATOR
# First moment
operator_moment_1= L/NX*np.linspace(L/NX, L, NX)
# Second moment
operator_moment_2= L/NX*np.square(np.linspace(L/NX, L, NX))
# Concatenate
observer = np.vstack((operator_moment_1, operator_moment_2))
###
mu = np.dot(observer,state_lw)

### KALMAN FILTER ############################################################

# Construction of the norm of the two spaces
standart_deviation = 0.0001
inv_norm_observation = 2*(NT/T)*(standart_deviation)**2*np.eye(2)

### STATE INIT A PRIORI
state_apriori = Gaussienne(x,0.7*cmax,1.2*imax,1.2*sigma)

### KALMAN FILTER
state_k =\
KalmanLW(L,NX,T,NT,\
mu,speed,\
state_apriori,np.eye(NX),\
observer,inv_norm_observation)


#### PLOTTING ##################################################################################

fig2, ax2 = plt.subplots()
#ax1.plot(x, state_init, 'k--', label = 'Model estimation')
ax2.plot(x, state_k[:,NT-1],'k:', label = 'Kalman')
ax2.plot(x, state_init,'k-', label = 'Initial')
ax2.plot(x, state_apriori,'r--', label = 'A priori')
legend = ax2.legend(loc='upper left', shadow=True, fontsize='x-large')
# Put a nicer background color on the legend.
legend.get_frame().set_facecolor('C0')
ax2.set(xlabel='time (s)', ylabel='convergence',
       title='Difference between solution and estimator')
ax2.grid()

plt.show()
