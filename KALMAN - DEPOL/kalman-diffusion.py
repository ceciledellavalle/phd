# -*- coding : utf-8 -*-

## KALMAN FILTER
## DEPOLYMERISATION ONLY
## TRANSPORT PLUS DIFFUSION epsilon

###################### 1D TRANSPORT EQUATION USING FINITE DIFFERENCES ##########################

### IMPORTATION
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import sys

### PHYSICAL PARAMETERS
L = 10.0  # Domain size
T = 10.0  # Integration time

### NUMERICAL PARAMETERS
NT = 100  # Number of time steps
NX = 100   # Initial number of grid points

### OBSERVER - MOMENT OPERATOR
# First moment
operator_moment_1= L/NX*np.linspace(L/NX, L, NX)
# Second moment
operator_moment_2= L/NX*np.square(np.linspace(L/NX, L, NX))
# Concatenate
observer = np.vstack((operator_moment_1, operator_moment_2))

### DATA GENERATION ############################################################################

### INITIAL CONDITION
x = np.linspace(0.0,L,NX)
sigma = L/10
gaussienne = L/10*np.exp(-(x-L/2)**2/(2*sigma**2)) # Gaussienne
gaussienne[NX-1]=0

### DEPOLIMERISATION PARAMETER b
b = 1.0 # Depolymerisation speed

### DIFFUSION PARAMETER e
e = 10.0 # Diffusion Speed


### DYNAMIC FLOW TRANSPORT EQUATION WITH DIFFUSION
def Flow(l,nx,t,nt,c,eps):
    # CFL condition
    CFL = c*(t/nt)*(nx/l)/2
    # Transport term
    trsp = - CFL*(np.diag(np.ones(NX-1),1)\
    -np.diag(np.ones(NX-1),-1))
    # Diffusion term
    diffu = - eps*CFL*(NX/L)/2*(-2*np.eye(NX,NX)\
    +np.diag(np.ones(NX-1),1)\
    +np.diag(np.ones(NX-1),-1))
    # Final flow computatiom
    flow_i = np.eye(NX,NX) + trsp + diffu
    flow_f = np.linalg.inv(flow_i)
    return flow_f

flow = Flow(L,NX,T,NT,b,e)


### VECTOR INITIALISATION
state = np.zeros((NX,NT)) # Saving the solution of the transport equation
state[:,0] = gaussienne.copy()
mu = np.zeros((2,NT)) # First and second order moment
mu[:,0] = np.dot(observer,state[:,0])

### DIRECT COMPUTATION OF THE SOLUTION
for i in range(0,NT-1):
    # Dynamic computation
    state[:,i+1] = np.dot(flow,state[:,i]) 
    # Calculation of the moment
    mu[:,i+1] = np.dot(observer,state[:,i+1])



### KALMAN FILTER #################################################################################

### INITIALISATION
# Construction of the observer (equal zero for the initial condition taken as parameter)
observer_kalman = np.vstack((operator_moment_1, np.zeros(NX), \
operator_moment_2, np.zeros(NX)))\
.reshape(2,2*(NX))
print(observer_kalman)

# Construction of the new dynamic (identity for the initial condition)
flow_kalman = np.concatenate((\
np.concatenate((flow,np.zeros((NX,NX))),axis=1),\
np.concatenate((np.zeros((NX,NX)),np.eye(NX)),axis=1)))

# Construction of the norm of the two spaces
standart_deviation = 0.00001
norm_observation = (T/NT)*(1/standart_deviation)**2*np.eye(2)
inv_norm_observation = (NT/T)*(standart_deviation)**2*np.eye(2)
#norm_state = (L/NX)*np.eye(NX)

# Initialisation of the state
state_init = 10*np.exp(-(x-L/3)**2/(2*sigma**2))
state_init[NX-1]=0
state_m = np.kron(np.ones(2),state_init)
state_p = state_m.copy()
state_kalman = np.zeros((2*NX,NT))
# Initialisation of the parameters
b_kalman = np.zeros(NT+1)
b_kalman[0] = 1.0
epsilon_kalman = np.zeros(NT+1)
epsilon_kalman[0] = 10

# Initialisation of the covariance matrix
covariance_operator_m = np.kron(np.ones((2,2)),np.eye(NX))
covariance_operator_p = covariance_operator_m.copy()
# Initialisation of the Pmatrix (transport and diffusion)
beta = 1
eta = 1
flow_be = Flow(L,NX,T,NT,b_kalman[0], epsilon_kalman[0])
T_init = (T/NT)*(NX/L)/2*(np.diag(np.ones(NX-1),1)\
    -np.diag(np.ones(NX-1),-1))
D_init = (T/NT)*(NX/L)/2*(NX/L)/2*(-2*np.eye(NX,NX)\
    +np.diag(np.ones(NX-1),1)\
    +np.diag(np.ones(NX-1),-1))
covariance_transport = beta**-1 * (flow_be - np.eye(NX))
covariance_diffusion = eta**-1 * D_init



### KALMAN FILTER
for k in range(0,NT):
    # Saving the solution
    state_kalman[:,k] = state_m.copy()

    ### CORRECTION
    # Covariance computation +
    interim_matrix = inv_norm_observation + observer_kalman.dot(covariance_operator_m).dot(observer_kalman.transpose())
    kalman_gain = covariance_operator_m.dot(observer_kalman.transpose()).dot(np.linalg.inv(interim_matrix))

    covariance_operator_p = (np.eye(2*NX) - kalman_gain.dot(observer_kalman)).dot(covariance_operator_m)\
    .dot(np.transpose((np.eye(2*NX) - kalman_gain.dot(observer_kalman)))) \
    + kalman_gain.dot(inv_norm_observation).dot(np.transpose(kalman_gain))

    # State correction computation +
    state_p = state_m + kalman_gain\
    .dot(mu[:,k]- np.dot(observer_kalman,state_m))

    # Parameters b correction computation +
    b_kalman[k+1] = b_kalman[k] \
    + (k+1)/b_kalman[k]*np.transpose(state_m[NX:2*NX]).dot(covariance_transport)\
    .dot(observer.transpose()).dot(norm_observation)\
    .dot(mu[:,k]- np.dot(observer_kalman,state_m))

    # Parameters epsilon correction computation +
    epsilon_kalman[k+1] = epsilon_kalman[k] \
    + (k+1)*b_kalman[k]*np.transpose(state_m[NX:2*NX]).dot(covariance_diffusion)\
    .dot(observer.transpose()).dot(norm_observation)\
    .dot(mu[:,k]- np.dot(observer_kalman,state_m))

    ### PREDICTION
    # Computation of the flow linked to the new parameters (b,e)
    flow_be = Flow(L,NX,T,NT,b_kalman[k+1], epsilon_kalman[k+1])
    # Construction of the new dynamic (identity for the initial condition)
    flow_kalman = np.concatenate((\
    np.concatenate((flow,np.zeros((NX,NX))),axis=1),\
    np.concatenate((np.zeros((NX,NX)),np.eye(NX)),axis=1)))

    # Covariance computation -
    covariance_operator_m = flow_kalman.dot(covariance_operator_p).dot(flow_kalman.transpose())

    # State prediction computation -
    state_m = np.dot(flow_kalman,state_p)

    # Parameters b correction computation -
    covariance_transport = flow_be.dot(covariance_transport)

    # Parameters epsilon correction computation -
    covariance_diffusion = flow_be.dot(covariance_diffusion)















#### PLOTTING ##################################################################################


### PLOTTING KALMAN
# Set up the figure, the axis, and the plot element we want to animate
fig_kalman = plt.figure()
ax_kalman = plt.axes(xlim=(0, L), ylim=(0, L/10+1))
line_kalman, = ax_kalman.plot([], [], lw=2)

# Initialization function: plot the background of each frame
def Initkalman():
    line_kalman.set_data([], [])
    return line_kalman,

# Animation function.  This is called sequentially
def Animatek(i):
    x = np.linspace(0, L, NX)
    y = state_kalman[NX:2*NX,i]
    line_kalman.set_data(x, y)
    return line_kalman,

animk = animation.FuncAnimation(fig_kalman, Animatek, \
init_func=Initkalman,\
frames=NT, interval=20, blit=True)


### Plotting b and epsilon
plt.subplot(2, 1, 1)
plt.plot(np.linspace(0.0, T,NT+1), b_kalman, 'o-')
plt.title('A tale of 2 subplots')
plt.ylabel('depolimerisation speed')

plt.subplot(2, 1, 2)
plt.plot(np.linspace(0.0, T,NT+1), epsilon_kalman, '.-')
plt.xlabel('time (s)')
plt.ylabel('diffusion speed')

plt.show()