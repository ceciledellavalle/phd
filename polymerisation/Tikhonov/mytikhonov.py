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

### TIKHONOV PARAMETER
noise = 0.001

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
fig0, ax0 = plt.subplots()
ax0.plot(np.linspace(0,T,NT), T/NT*NX/L*speed, 'b--', label='speed')
#ax0.plot(np.linspace(0,L,NX), state_init, 'r--', label='speed')
legend = ax0.legend(loc='lower right', shadow=True, fontsize='x-large')
ax0.set(xlabel='time (s)', ylabel='cfl',
       title='Becker Döring - total speed equivalency')

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

moment0 = observer0.dot(state) + noise*np.random.rand(NT)
print("level of noise {}".format(\
np.linalg.norm(moment0-observer0.dot(state))
/np.linalg.norm(observer0.dot(state))))

# anim = PlotDymamicSolution(1.1*L,1.1*cmax,\
# np.linspace(0,L,NX),state,
# NT,np.linspace(0,T,NT))

################################################################################
## TIKHONOV
################################################################################


### REGULARISATION ALPHA
alpha = 0.004
phi = np.zeros((NX,NT))
for i in range(0,NX):
    for j in range(0,NT):
        nl = int(upsilon[j])
        ### OPERATOR PSI
        phi[nl:,j]=L/NX*np.ones(NX-nl)
        ### REGULARIZA TION
        if i == nl:
            phi[i,j] += -alpha*speed[j]


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


##############################################################################
### GRADIENT TESTING #######)##################################################
#Initialisation
npoint = 50
beta_liste = np.exp(np.linspace(0.0,-20.0,npoint))
gradJ0 = np.zeros(npoint)
gradJ1 = np.zeros(npoint)
epsilon = np.zeros(npoint)
zeta = np.random.rand(NX)
#For decreasing beta
for j, beta in enumerate(beta_liste):
    zeta_1 = np.random.rand(NX)
    norm_zeta_1 = (L/NX)*np.linalg.norm(zeta_1)
    zeta_1 = zeta_1/norm_zeta_1
    # Compute difference
    gradJ0[j] = Rosen(zeta+beta*zeta_1,\
    phi.T,T,NT,L,NX,moment0)\
    - Rosen(zeta,\
    phi.T,T,NT,L,NX,moment0)
    gradJ0[j] = gradJ0[j]/beta
    # Compute grad
    gradJ1v = Rosen_der(zeta,\
    phi.T,T,NT,L,NX,moment0)
    gradJ1[j] = gradJ1v.dot(zeta_1)
    # Compute the error between difference and grad
    epsilon[j] = abs(gradJ0[j]-gradJ1[j])

### PLOTS !!
print(epsilon)
# plt.loglog()
# plt.xlabel('beta')
# plt.plot(beta_liste,gradJ0, label ="difference")
# plt.plot(beta_liste,gradJ1, label = "gradient")
# plt.legend()

##############################################################################
##############################################################################




### CONTROL OF ROSEN SOLUTION
# Create plots with pre-defined labels.
# moment0_sim = matrix_psi.dot(state_init)
# fig1, ax1 = plt.subplots()
# ax1.plot(np.linspace(0,T,NT), moment0, 'b--', label='C state')
# ax1.plot(np.linspace(0,T,NT), moment0_sim, 'b-', label='Psi y')
# legend = ax1.legend(loc='lower right', shadow=True, fontsize='x-large')
# ax1.set(xlabel='time (s)', ylabel='moment',\
#          title='Evolution of the 0 order moment')



##############################################################################
### MINIMISATION
# state_apriori = Gaussienne(x,0.9*cmax,1.1*imax,sigma)
# res = minimize(Rosen, state_apriori, args=(m_psi,m_reg,upsilon,moment0),\
#  method='Nelder-Mead', tol=1e-6)

def Box(y,a,b):
    y[y<a]=a
    y[y>b]=b
    return y

def Soft(y,alpha):
    f = [0 if abs(x)<alpha else x+ math.copysign(alpha,x) for x in y]
    print
    return f

pas_grad = 2/np.linalg.norm(phi)**2

################
y = np.zeros(NX)
for iter in range(0,1000):
    y -=pas_grad*Rosen_der(y,\
    phi.T,T,NT,L,NX,moment0)
    y =Box(y,0,1)

fig1, ax1 = plt.subplots()
ax1.plot(np.linspace(0,L,NX), y, 'b-', label='gradient descent')
ax1.plot(np.linspace(0,L,NX), state_init, 'r--', label='initial state')
legend = ax1.legend(loc='upper left', shadow=True, fontsize='x-large')
ax1.set(xlabel='taille (mm)', ylabel='concentration',
       title='Reconstruction of initial condition under noisy moment (noise 11%)')

################################################################################
################################################################################
################################################################################
plt.show()
