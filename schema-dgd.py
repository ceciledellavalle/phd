# -*- coding : utf-8 -*-

###################### 1D TRANSPORT EQUATION USING FINITE DIFFERENCES ##########################

### IMPORTATION
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import sys


### DOMAIN SIZE
L = 100.0  # Domain size
T = 100.0  # Integration time

### NUMERICAL PARAMETERS
NT = 100  # Number of time steps
NX = 100   # Initial number of grid points

### PHYSICAL PARAMETERS
# Polymerisation speed a =
a = 0.00*np.ones(NX) # Polymerisation speed
a[NX-1]=0
# Depolimerisation speed b =
b = 1*np.ones(NX)  # Depolymerisation speed
# Monomers concentration
c0 = 100
# Polymers repartition
sigma = L/10

### CFL TEST
CFL = T/NT*NX/L*np.amax(c0*a+b)
if CFL>1:
    NT = math.ceil(T*NX/L*np.amax(c0*a+b))
    print("The new time step is = {}.".format(T/NT))



### MOMENT OPERATOR ################################################################################

# First order
moment = L/NX*np.linspace(L/NX, L, NX)
moment[0] = 1/2*moment[0]
moment[NX-1] = 1/2*moment[NX-1]

# Second order
moment2 = L/NX*np.square(np.linspace(L/NX, L, NX))
moment2[0] = 1/2*moment2[0]
moment2[NX-1] = 1/2*moment2[NX-1]

### FLOW COMPUTATION ###############################################################################

def Flow(a,b,T,NT,L,NX,co):
    alpha = T/NT*NX/L
    flow = np.eye(NX,NX) \
    - alpha*np.diag(co*a+b) \
    + alpha*np.diag(co*a[:-1],-1)\
    + alpha*np.diag(b[1:],1)
    return flow

### LS SOLUTION ITER ###############################################################################

### INITIAL CONDITION
# System state u
x = np.linspace(0.0,L,NX)
state = np.zeros((NX,NT))
state[:,0] = sigma*np.exp(-(x-L/2)**2/(2*sigma**2)) # Gaussienne
state[0,0] = 0
state[NX-1,0] = 0
# Monomere concentration
cmono = np.zeros(NT)
cmono[0] = c0
# Computation of the constant total mass
masse_rho = cmono[0] + moment.dot(state[:,0])


### EVOLUTION COMPUTATION
for k in range(0,NT-1):
    # State evolution
    state[:,k+1] = Flow(a,b,T,NT,L,NX,cmono[k]).dot(state[:,k])
    # Monomer evolution
    cmono[k+1] = masse_rho - moment.dot(state[:,k+1])

### WE PLOT #########################################################################################

### PLOTTING SOLUTION OF 1D TRANSPORT EQUATION
# Set up the figure, the axis, and the plot element we want to animate
fig_state = plt.figure()
ax_state = plt.axes(xlim=(0, L), ylim=(0, L/10+1))
line_state, = ax_state.plot([], [], lw=2)

# Initialization function: plot the background of each frame
def Initstate():
    line_state.set_data([], [])
    return line_state,

# Animation function.  This is called sequentially
def Animate1DT(i):
    x = np.linspace(0, L, NX)
    y = state[:,i]
    line_state.set_data(x, y)
    return line_state,

anim = animation.FuncAnimation(fig_state, Animate1DT, \
init_func=Initstate,\
frames=NT-1, interval=50, blit=True)

### PLOTING MONOMERS CONCENTRATION
fig_monomers = plt.figure()
ax_monomers = plt.axes(xlim=(0, T), ylim=(0,masse_rho))
ax_monomers.set_xlabel("Time")
ax_monomers.set_ylabel("Concentration")
ax_monomers.plot(np.linspace(0.0,T,NT),cmono,'k--', label='Monomers')
legend = ax_monomers.legend(loc='upper right', shadow=True, fontsize='x-large')
# Put a nicer background color on the legend.
legend.get_frame().set_facecolor('C0')


### PLOTTING SECOND ORDER
mu2 = moment2.dot(state)
fig_mu2 = plt.figure()
ax_mu2 = plt.axes(xlim=(0, T), ylim=(0,np.amax(mu2)))
ax_mu2.set_xlabel("Time")
ax_mu2.set_ylabel(" ")
ax_mu2.plot(np.linspace(0.0,T,NT),mu2,'k:', label='Second order')
legend = ax_mu2.legend(loc='upper right', shadow=True, fontsize='x-large')
# Put a nicer background color on the legend.
legend.get_frame().set_facecolor('C1')


plt.show()