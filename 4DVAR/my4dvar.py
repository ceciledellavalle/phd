# -*- coding : utf-8 -*-

###################### 1D TRANSPORT EQUATION USING FINITE DIFFERENCES ##########################


### IMPORTATION
import numpy as np
import matplotlib.pyplot as plt
import math
from fractions import Fraction
import sys


### PHYSICAL PARAMETERS
b = 0.2  # Depolymerisation speed
L = 1.0  # Domain size
T = 5.0  # Integration time

### NUMERICAL PARAMETERS
NT = 50 # Number of time steps
NX = 50 # Initial number of grid points
step_save = 10 # Numbers of saving (to plot the solution)

### GRADIENT PARAMETER
iteration_max = 10000
gamma = 100000000
alpha = 1
standart_deviation = 1
# Test if the gradient is well computed
test_gradient = False

### OBSERVER - MOMENT OPERATOR
# First moment
operator_moment_1 = (L/NX)*np.linspace(0, L, NX)
# Second moment
operator_moment_2= (L/NX)*np.square(np.linspace(0, L, NX))
# Concatenate
observer = np.vstack((operator_moment_1, operator_moment_2))
observer_transpose = np.transpose(observer)


### DYNAMIC FLOW PHI
CFL = b*(T/NT)*(NX/L)
flow = np.eye(NX,NX) + CFL*(-np.eye(NX,NX) + np.diag(np.ones(NX-1),1))
flow_transpose = np.eye(NX,NX) + CFL*(-np.eye(NX,NX) + np.diag(np.ones(NX-1),-1))


### DATA GENERATION ############################################################################

### INITIAL CONDITION
x = np.linspace(0.0,L,NX+1)
sigma = L/10
gaussienne = (sigma*np.sqrt(2*np.pi))**-1*np.exp(-(x-L/2)**2/(2*sigma**2)) # Gaussienne
gaussienne[NX]=0

### VECTOR INITIALISATION
state = np.zeros((NX+1,NT+1)) # Saving the solution of the transport equation
state[:,0] = gaussienne.copy() # Initial condition set as a gaussian
step_register = int((NT+1)/step_save)
mu = np.zeros((2,NT+1)) # First and second order moment

### DIRECT COMPUTATION OF THE SOLUTION
for i in range(0,NT):
    # Calculation of the moment
    mu[:,i] = np.dot(observer,state[:-1,i])
    # Dynamic computation
    state[:-1,i+1] = np.dot(flow,state[:-1,i]) 


### 4DVAR #######################################################################################

### INITIALISATION  #############################################################################

# Construction of the norm of the two spaces
norm_observation = (T/NT)*(1/standart_deviation)**2*np.eye(2)
inv_norm_observation = (standart_deviation)**2/(T/NT)*np.eye(2)
norm_state = (L/NX)*np.eye(NX)
inv_norm_state = (NX/L)*np.eye(NX)

# State Initialisation
gaussienne_d = (sigma*np.sqrt(2*np.pi))**-1*np.exp(-(x-L/3)**2/(2*sigma**2))
gaussienne_d[NX]=0
double_ic = np.cos(4*np.pi*x/L+np.pi)+ np.ones(NX+1)
double_ic[NX] =0
# Adjoint state 
adjoint_state = np.zeros((NX+1,NT+1))
# State
state_4dvar = np.zeros((NX+1,NT+1)) # current state
# Initial state a priori
state_init = np.zeros((NX+1,iteration_max)) # Initial condition knowing n obervations
#state_init[:,0] = gaussienne_d.copy()
# Initial state deviation
state_4dvar[:-1,0] = state_init[:-1,0].copy()
zeta = np.zeros((NX+1,iteration_max))

### PLOTTING SOLUTION  #############################################################################


### PARAMETRE DE RELAXATION #########################################################################
# Calculation of the quadratic matrix A of the function J = 1/2(Azeta,zeta) - (b,zeta)
A_quadratique = alpha*norm_state.copy()
A_inter = gamma*observer_transpose.dot(norm_observation).dot(observer)
A_quadratique += A_inter
for i in range(1,NT+1):
    A_inter = flow_transpose.dot(A_inter).dot(flow)
    A_quadratique += A_inter

# Calculation of the quadratic matrix b of the function J = 1/2(Azeta,zeta) - (b,zeta)
# If test = True then computation of the matrix b 
# in order to compute explicitely differential J
if test_gradient:
    state_apriori = np.zeros((NX,NT+1))
    state_apriori[:,0] = state_init[:-1,0].copy()
    b_linear = gamma*observer_transpose.dot(norm_observation)\
    .dot(mu[:,0]-observer.dot(state_apriori[:,0]))
    for j in range(1,NT+1):
        state_apriori[:,j]=flow.dot(state_apriori[:,j-1])
        b_inter = gamma*observer_transpose.dot(norm_observation)\
        .dot(mu[:,j]-observer.dot(state_apriori[:,j]))
        for k in range(1,j+1):
            b_inter = flow_transpose.dot(b_inter)
        b_linear += b_inter

# Eigenvalue
matrix_eigen = np.linalg.eig(A_quadratique)
eigenvalue_NX = matrix_eigen[0][0]
eigenvalue_0 = matrix_eigen[0][-1]

#Gradient descent parameters
relaxation1 = 2*eigenvalue_0/(eigenvalue_NX**2)
relaxation2 = 2/(eigenvalue_NX+eigenvalue_0)
relaxation = max(relaxation1,relaxation2)
print("le coefficient de relaxation est = {}".format(relaxation))



### GRADIENT DESCENT  ############################################################################

epsilon = np.zeros(iteration_max)
iteration = 0

while (iteration<iteration_max-1):

    # Comnputation of solution knowing the initial condition with k observation
    for i in range(0,NT):
        state_4dvar[:,i+1] = np.dot(flow,state_4dvar[:,i])
        
    # Computation of the adjoint state
    adjoint_state[:, NT] = np.zeros(NX+1)
    for ii in range(NT,0,-1):
        adjoint_state[:,ii-1] = np.dot(flow_transpose, adjoint_state[:,ii])\
        + observer_transpose.dot(norm_observation)\
        .dot(mu[:,ii-1]-np.dot(observer,state_4dvar[:,ii-1]))
        
    # Incrementing the indicator
    iteration+=1

    # Computation of the gradient
    zeta[:,iteration] = zeta[:,iteration-1] \
    -relaxation*(alpha*norm_state.dot(zeta[:,iteration-1]) - gamma*adjoint_state[:,0])

    # Testing : is the gradient correct
    if test_gradient:
        # Precision
        beta = 0.001
        # Diffrential direction
        delta_zeta = beta*np.random.rand(NX+1)
        # Computation by Taylor
        gradJ_diff = np.transpose(np.dot(A_quadratique,zeta[:,iteration-1])).dot(delta_zeta[:]) \
        -b_linear.dot(delta_zeta[:])
        # Computation by adjoint
        grad_adjoint =np.transpose(alpha*norm_state.dot(zeta[:,iteration-1]) \
        - gamma*adjoint_state[:,0])\
        .dot(delta_zeta[:])
        smallo = abs(gradJ_diff - grad_adjoint)
        print(smallo)
        if smallo > beta:
            print("Sorry, the computation of the gradient is false.")
    
    # Computation of epsilon for testing or plotting purpose
    epsilon[iteration-1] = np.transpose(alpha*zeta[:-1,iteration-1]\
    - gamma*inv_norm_state.dot(adjoint_state[:-1,0]))\
    .dot(norm_state)\
    .dot(alpha*zeta[:-1,iteration-1]\
    - gamma*inv_norm_state.dot(adjoint_state[:-1,0]))
 
    # Computation of the initial state
    state_init[:-1,iteration] = state_init[:-1,0] + zeta[:-1,iteration]
    state_init[NX,iteration] = 0
    
    # Re initiatisation of the first step of current state
    state_4dvar[:-1,0] = state_init[:-1,iteration].copy()


### PLOTTING SOLUTION  #############################################################################

    
plt.xlabel(u'$x$', fontsize=5)
plt.ylabel(u'$y$', fontsize=5, rotation=0)
plt.title("CFL = {0}, alpha/gamma = {1}, iteration ={2}"\
.format(round(CFL,2),alpha/gamma,iteration_max))
p0 = plt.plot(x, state_init[:,0], label= "a priori")
p1 = plt.plot(x, state_init[:,iteration_max-1], label= "estimateur 4dvar")
p2 = plt.plot(x, gaussienne, label= "condition initiale")
plt.legend()
plt.show()
