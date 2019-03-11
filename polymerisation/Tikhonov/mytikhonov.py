# -*- coding : utf-8 -*-

## TIKHONOV REGULATION


### IMPORTATION
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import sys
# optimization function
from scipy.optimize import minimize
# dense to sparse
from scipy.sparse import csr_matrix
##
from bdschemek import BeckerDoringScheme
from bdschemek import SpeedComputation
from plotdynamic import PlotDymamicSolution

### PHYSICAL PARAMETERS
b = 3.0  # Depolymerisation speed
a = 1.0
c0 = 0.0
L = 20.0  # Domain size
T = 10.0  # Integration time

### NUMERICAL PARAMETERS
NT = 150  # Number of time steps
NX = 100   # Initial number of grid points



### BECKER DORING ###########################################################

### INITIAL CONDITION
x = np.linspace(0.0,L,NX)
cmax = 0.01 # Maximum concentration of polymers
imax = L/2
sigma = 2
def Gaussienne(x,c,i,s):
    gaussienne = c*np.exp(-(x-i)**2/(2*s**2)) # Gaussienne
    Ncrop = math.ceil(NX*0.2)
    gaussienne[0:Ncrop] = np.zeros(Ncrop)
    gaussienne[NX-Ncrop:] = np.zeros(Ncrop)
    return gaussienne


### VECTOR INITIALISATION
# Initial condition set as a gaussian
state_init = Gaussienne(x,cmax,imax,sigma)

### SOLUTION OF BDSCHEME
state_bd = BeckerDoringScheme(L,NX,T,NT,a,b,c0,state_init)

### Computation of equivalent speed
speed = SpeedComputation(L,NX,T,NT,a,b,c0,state_bd)

### Computation of upsilon
upsilon_c = np.abs(T/NT*np.cumsum(speed))
# interpolation
upsilon = np.floor((NX-1)/L*upsilon_c)
upsilon = np.minimum(upsilon, NX-1)


### PLOTs
# fig0, ax0 = plt.subplots()
# ax0.plot(np.linspace(0,T,NT), T/NT*NX/L*speed, 'b--', label='speed')
# #ax0.plot(np.linspace(0,L,NX), state_init, 'r--', label='speed')
# legend = ax0.legend(loc='lower right', shadow=True, fontsize='x-large')
# ax0.set(xlabel='time (s)', ylabel='cfl',
#        title='Becker Döring - total speed equivalency')

# anim_bd = PlotDymamicSolution(1.1*L,1.1*cmax,\
# np.linspace(0,L,NX),state_bd,
# NT,np.linspace(0,T,NT))

################################################################################
## DATA GENERATION
################################################################################

### EXPLICIT EULER Scheme
state = np.zeros((NX,NT))
state[:,0] = state_init
### DYNAMIC FLOW PHI
for i in range(0,NT-1):
    cfl = abs(speed[i])*T/NT*NX/L
    flow = np.eye(NX,NX) + cfl*(-np.eye(NX,NX) + np.diag(np.ones(NX-1),1))
    state[:,i+1] = flow.dot(state[:,i])

### OBSERVER
observer0 = L/NX*np.ones(NX)
observer0[0] = observer0[0]/2
observer0[NX-1] = observer0[NX-1]/2

moment0 = observer0.dot(state)

# anim = PlotDymamicSolution(1.1*L,1.1*cmax,\
# np.linspace(0,L,NX),state,
# NT,np.linspace(0,T,NT))

################################################################################
## TIKHONOV
################################################################################

### REGULARISATION ALPHA
def Phi0(alpha,L,NX,T,NT,upsilon,speed):
    phi = np.zeros((NX,NT))
    for i in range(0,NX):
        for j in range(0,NT):
            nl = int(upsilon[j])
            ### OPERATOR PSI
            phi[nl:,j]=L/NX*np.ones(NX-nl)
            ### REGULARIZA TION
            if i == nl:
                phi[i,j] += -alpha*speed[j]
    return phi

### DEFINITION OF ROSEN
def Rosen(y,matrix0,\
T,NT,L,NX,moment0):
    # compute phi(y)
    z= matrix0.dot(y)
    # compute the error
    rosen = 1/2*T/NT*(z-moment0).T.dot(z-moment0)
    return rosen

### DEFINITION OF GRAD ROSEN
def Rosen_der(y,matrix0,\
T,NT,L,NX,moment0):
    #############################
    ## Calcul de Ay-b
    # add the psi part
    z= matrix0.dot(y)-moment0
    # retrieve moment of order 0
    ###############################
    ## Calcul de transpose A
    grad = matrix0.T.dot(z)
    # Scalar product
    return T/NT*grad

def Error(y1,y2):
    e = L/NX*np.linalg.norm(y1-y2,2)
    return e


##############################################################################
##############################################################################
### MINIMISATION
def Box(y,a,b):
    y[y<a]=a
    y[y>b]=b
    return y

def Soft(y,alpha):
    f = [0 if abs(x)<alpha else x+ math.copysign(alpha,x) for x in y]
    return f


##############################################################################
##############################################################################
# ERROR REGARDING EPSILON

Npoint= 10

epsilon = np.zeros(Npoint)
alpha = np.zeros(Npoint)
error = np.zeros(Npoint)
i = 0
for lnoise in np.logspace(-6,6,Npoint, endpoint=False):
    noise = lnoise*np.random.rand(NT)
    ze = moment0 + noise
    epsilon[i] = T/NT*np.linalg.norm(noise,2)
    alpha[i] = epsilon[i]
    y = np.zeros(NX)
    phi = Phi0(alpha[i],L,NX,T,NT,upsilon,speed)
    pas_grad = 2/np.linalg.norm(phi)**2
    for iter in range(0,1000):
        y -=pas_grad*Rosen_der(y,\
        phi.T,T,NT,L,NX,moment0)
    error[i] = Error(y, state_init)
    # iteration on alpha and epsilon
    i+=1

fig1, ax1 = plt.subplots()
ax1.loglog(epsilon, error, 'r+')
ax1.set(xlabel='epsilon', ylabel='erreur',
       title='Error for alpha optimal')


##############################################################################
##############################################################################
# ERROR REGARDING ALPHA


Npoint= 11
lnoise = 0.001
noise = lnoise*np.random.rand(NT)
ze = moment0 + noise
epsilon = T/NT*np.linalg.norm(noise)

alpha = np.logspace(-4,10,Npoint, endpoint=False)
error = np.zeros(Npoint)
i = 0
for a in alpha:
    y = np.zeros(NX)
    phi = Phi0(a,L,NX,T,NT,upsilon,speed)
    pas_grad = 2/np.linalg.norm(phi)**2
    for iter in range(0,1000):
        y -=pas_grad*Rosen_der(y,\
        phi.T,T,NT,L,NX,moment0)
    error[i] = Error(y, state_init)
    # iteration on alpha and epsilon
    i+=1

fig1, ax1 = plt.subplots()
ax1.loglog(alpha, error, 'g+')
ax1.set(xlabel='alpha', ylabel='erreur',
       title='Error for different value of alpha')


################################################################################
################################################################################
################################################################################
plt.show()

# fig1, ax1 = plt.subplots()
# ax1.plot(np.linspace(0,L,NX), y, 'b-', label='gradient descent')
# ax1.plot(np.linspace(0,L,NX), state_init, 'r--', label='initial state')
# legend = ax1.legend(loc='upper left', shadow=True, fontsize='x-large')
# ax1.set(xlabel='taille (mm)', ylabel='concentration',
#        title='Reconstruction of initial condition under noisy moment (noise 11%)')
