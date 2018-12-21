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

### PHYSICAL PARAMETERS
b = 2.0  # Depolymerisation speed
L = 10.0  # Domain size
T = 5.0  # Integration time

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

### DYNAMIC FLOW PHI
CFL = b*(T/NT)*(NX/L)
print(CFL)
flow = np.eye(NX,NX) + CFL*(-np.eye(NX,NX) + np.diag(np.ones(NX-1),1))

### DATA GENERATION ############################################################################

### INITIAL CONDITION
x = np.linspace(0.0,L,NX)
sigma = L/10
gaussienne = 10*np.exp(-(x-L/2)**2/(2*sigma**2)) # Gaussienne
gaussienne[NX-1]=0

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

# Construction of the new dynamic (identity for the initial condition)
flow_kalman = np.concatenate((\
np.concatenate((flow,np.zeros((NX,NX))),axis=1),\
np.concatenate((np.zeros((NX,NX)),np.eye(NX)),axis=1)))

# Construction of the norm of the two spaces
standart_deviation = 0.00001
norm_observation = (T/NT)*(1/standart_deviation)**2*np.eye(2)
inv_norm_observation = (NT/T)*(standart_deviation)**2*np.eye(2)
#norm_state = (L/NX)*np.eye(NX)

# Initialisation of the covariance matrix
covariance_operator_m = np.kron(np.ones((2,2)),np.eye(NX))
covariance_operator_p = covariance_operator_m.copy()

# Initialisation of the state
state_init = 10*np.exp(-(x-L/3)**2/(2*sigma**2))
state_init[NX-1]=0
state_m = np.kron(np.ones(2),state_init)
state_p = state_m.copy()
state_kalman = np.zeros((2*NX,NT))

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

    ### PREDICTION
    # Covariance computation -
    covariance_operator_m = flow_kalman.dot(covariance_operator_p).dot(flow_kalman.transpose())

    # State prediction computation -
    state_m = np.dot(flow_kalman,state_p)


#### PLOTTING ##################################################################################


### PLOTTING KALMAN
# Set up the figure, the axis, and the plot element we want to animate
fig_kalman = plt.figure()
ax_kalman = plt.axes(xlim=(0, L), ylim=(-np.amax(state_kalman), np.amax(state_kalman)))
line_kalman, = ax_kalman.plot([], [], lw=2)

# Initialization function: plot the background of each frame
def Initkalman():
    line_kalman.set_data([], [])
    return line_kalman,

# Animation function.  This is called sequentially
def Animatek(i):
    x = np.linspace(0, L, NX)
    y = state_kalman[NX:,i]
    line_kalman.set_data(x, y)
    return line_kalman,

animk = animation.FuncAnimation(fig_kalman, Animatek, \
init_func=Initkalman,\
frames=NT, interval=20, blit=True)



### SHOW PLOTS
plt.xlabel(u'$x$', fontsize=10)
plt.ylabel(u'$y$', fontsize=10, rotation=0)
plt.show()

### PLOTTING SOLUTION OF 1D TRANSPORT EQUATION
# Set up the figure, the axis, and the plot element we want to animate
fig_state = plt.figure()
ax_state = plt.axes(xlim=(0, L), ylim=(0, 10+1))
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
frames=NT, interval=20, blit=True)


### PLOTTING NOISE GROWING ON OBSERVATIONS
# Set up the figure, the axis, and the plot element we want to animate
fig_noise = plt.figure()
ax_noise = plt.axes(xlim=(0, T), ylim=(-np.max(mu[0,:]),np.max(mu[0,:])))
line_noise, = ax_noise.plot([], [], lw=2)
noise_text = ax_noise.text(0.02, 0.95, '', transform=ax_noise.transAxes)


# Initialization function: plot the background of each frame
def Initnoise1():
    line_noise.set_data([], [])
    noise_text.set_text('')
    return line_noise, noise_text

# Animation function.  This is called sequentially
def Animatenoise1(i):
    x = np.linspace(0, T, NT)
    # generate noise vector
    noise_level = i*0.05*(T/NT)*1/2*np.sqrt(np.linalg.norm(mu[0,:]))
    noise = noise_level*np.random.rand(NT)
    y = mu[0,:] + noise
    line_noise.set_data(x, y)
    noise_text.set_text("Noise level = {} %".format(round(i*0.05,3)))
    return line_noise, noise_text

#anim2 = animation.FuncAnimation(fig_noise, Animatenoise1, \
#init_func=Initnoise1,\
#frames=NT, interval=20, blit=True)

plt.xlabel('x')
plt.ylabel('c')
plt.axes(xlim=(0, L), ylim=(-2,10))
#plit.ylabel("difference")
p0 = plt.plot(np.linspace(0, L, NX),state_kalman[NX:,0], label ="apriori")
p1 = plt.plot(np.linspace(0, L, NX),state_kalman[NX:,NT-1], label = "optimal")
plt.title("Condition initiale h={0},dt={1}".format(L/NX,T/NT))
plt.legend()
plt.show()