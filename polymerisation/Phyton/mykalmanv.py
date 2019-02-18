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
c0 = 1
L = 20.0  # Domain size
T = 5.0  # Integration time

### NUMERICAL PARAMETERS
NT = 100  # Number of time steps
NX = 100   # Initial number of grid points

### BECKER DORING ###########################################################

### INITIAL CONDITION
x = np.linspace(0.0,L,NX)
cmax = 0.1 # Maximum concentration of polymers
imax = 2*L/3
sigma = L/20
def Gaussienne(x,c,i,s):
    gaussienne = c*np.exp(-(x-i)**2/(2*s**2)) # Gaussienne
    return gaussienne


### VECTOR INITIALISATION
# Initial condition set as a gaussian
state_init = Gaussienne(x,cmax,imax,sigma)

### SOLUTION OF BDSCHEME
state_bd = BeckerDoringScheme(L,NX,T,NT,a,b,c0,state_init)

###PLOTTING
#anim_bd = PlotDymamicSolution(1.1*L,1.1*cmax,\
#np.linspace(0,L,NX),state,
#NT,np.linspace(0,T,NT))

speed = SpeedComputation(L,NX,T,NT,a,b,c0,state_bd)

### PLOTs
# Create plots with pre-defined labels.
#fig0, ax0 = plt.subplots()
#ax0.plot(np.linspace(0,T,NT), T/NT*NX/L*speed, 'k--', label='speed')
#ax0.plot(np.linspace(0,T,NT), delta_time, 'k:', label='delta t')
#legend = ax0.legend(loc='upper right', shadow=True, fontsize='x-large')

### LAX WENDROFF ###########################################################

state_lw = LaxWendroffScheme(L,NX,T,NT,speed,state_init)

### PLOTs
anim_lw = PlotDymamicSolution(1.1*L,1.1*cmax,\
np.linspace(0,L,NX),state_lw,
NT,np.linspace(0,T,NT))
plt.show()
sys.exit()

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
norm_state = (L/NX)*np.eye(NX)
inv_norm_state = (NX/2)*np.eye(NX)
norm_observation = 1/2*(T/NT)*(1/standart_deviation)**2*np.eye(2)
inv_norm_observation = 2*(NT/T)*(standart_deviation)**2*np.eye(2)
# Combine the norm in one list
norm = [norm_state, inv_norm_state,\
norm_observation, inv_norm_observation]

### STATE INIT A PRIORI
state_apriori = Gaussienne(x,0.7*cmax,0.5*imax,0.8*sigma)

### KALMAN FILTER
#state_k = KalmanAdaptedTimeScale(L,NX,T,NT,\
#mu,speed,\
#state_apriori,np.diag(np.ones(NX-1),1)),observer,norm)

state_k =\
KalmanLW(L,NX,T,NT,\
mu,speed,\
state_apriori,np.eye(NX),\
observer,inv_norm_observation)

anim_k = PlotDymamicSolution(1.1*L,1.1*cmax,\
np.linspace(0,L,NX),state_k,
NT,np.linspace(0,T,NT))

#### PLOTTING ##################################################################################

fig2, ax2 = plt.subplots()
#ax1.plot(x, state_init, 'k--', label = 'Model estimation')
ax2.plot(x, state_k[:,NT-1],'k:', label = 'Solution')
legend = ax2.legend(loc='upper right', shadow=True, fontsize='x-large')
# Put a nicer background color on the legend.
legend.get_frame().set_facecolor('C0')
ax2.set(xlabel='time (s)', ylabel='convergence',
       title='Difference between solution and estimator')
ax2.grid()

plt.show()
