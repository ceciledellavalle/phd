# -*- coding : utf-8 -*-

###################### 1D TRANSPORT EQUATION USING FINITE DIFFERENCES ##########################
################################################################################################
######################  EULER IMPLICIT UPSTREAM DOWNSTREAM SCHEME     ##########################

### IMPORTATION
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import sys

### NUMERICAL PARAMETERS
NT = 100  # Number of time steps
NX = 100  # Initial number of grid points


### PHYSICAL PARAMETERS
a = 0.01*np.ones(NX) # Polymerisation speed
a[NX-1] = 0
b = 0.5*np.ones(NX)  # Depolymerisation speed
b[0] = 0
L = 10.0  # Domain size
T = 5.0  # Integration time
alpha = NX/L*T/NT
c0 = 100 # Concentration of monomers in the system

### WE INITIALIZE #####################################################################################

### INTERMEDIATE MATRIX INITIALISATION
A = alpha*(np.diag(a[1:]) - np.diag(a[1:-1],-1))
B = alpha*(np.diag(b[1:]) - np.diag(b[1:-1],1))
moment = L/NX*np.linspace(L/NX, L, NX)

### INITIAL CONDITION
x = np.linspace(L/NX,L,NX-1)
# System state u
sigma = L/10
state = np.zeros((NX,NT+1))
state[1:,0] = sigma*np.exp(-(x-L/2)**2/(2*sigma**2)) # Gaussienne
state[0,0] = c0
# Computation of the constant total mass
masse_rho = moment.dot(state[:,0])

### WE COMPUTE ########################################################################################

### ITERATION PARAMETERS
iter_max = 100
s  = 0
et = L/NX*np.linspace(2*L/NX, L, NX-1)
ds = np.zeros(NX-1)

for k in range(0,NT):

    # Initialisation of increment delta
    delta = 100*np.ones(NX-1)
    s = 0
    # Set c(s=0) = ck
    state[:,k+1] = state[:,k].copy()

    while (s<iter_max)&(np.amax(delta)>0.001):

        # Intermediate matrix dF = [1,et];[ds,Ts]
        Ts = np.eye(NX-1,NX-1) + state[0,k+1]*A + B
        Tsinv = np.linalg.inv(Ts)
        ds[1:] = a[2:]*state[2:,k+1]-a[1:-1]*state[1:-1,k+1]
        ds[0] = a[1]*state[1,k+1] - 2*a[0]*state[0,k+1]
        ds = NX/L*T/NT*ds
        Fhat = -(state[1:,k+1] - state[1:,k] + (state[0,k+1]*A+B).dot(state[1:,k+1]))
        Fhat[0] = Fhat[0] + alpha*a[0]*state[0,k+1]**2

        # Variation of the monomers
        delta1 = masse_rho - moment.dot(state[:,k+1]) - et.dot(Tsinv).dot(Fhat)
        delta1 = delta1/(L/NX-et.dot(Tsinv).dot(ds))
        # variations of the polymers
        delta = Tsinv.dot(Fhat) - delta1*Tsinv.dot(ds)

        # Computation of the solution
        state[0,k+1] = state[0,k+1] + delta1
        state[1:,k+1] = state[1:,k+1] + delta

        # Incrementing iteration step
        s=s+1

                

### WE PLOT #########################################################################################

### PLOTTING SOLUTION OF 1D TRANSPORT EQUATION
# Set up the figure, the axis, and the plot element we want to animate
fig_state = plt.figure()
ax_state = plt.axes(xlim=(0, L), ylim=(-sigma, sigma))
line_state, = ax_state.plot([], [], lw=2)

# Initialization function: plot the background of each frame
def Initstate():
    line_state.set_data([], [])
    return line_state,

# Animation function.  This is called sequentially
def Animate1DT(i):
    x = np.linspace(L/NX, L, NX-1)
    y = state[1:,i]
    line_state.set_data(x, y)
    return line_state,

anim = animation.FuncAnimation(fig_state, Animate1DT, \
init_func=Initstate,\
frames=NT, interval=50, blit=True)

plt.show()
