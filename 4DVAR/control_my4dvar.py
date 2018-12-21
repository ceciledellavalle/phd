# -*- coding : utf-8 -*-

## CALCUL DE LA FONCTIONNELLE J = 1/2<Au,u> + bu
## CALCUL DU GRADIENT DJ = Au + b
## CALCUL DU GRADIENT PAR LA METHODE DE L'ADJOINT

## COMPARAISON DES TROIS METHODES

###################### 1D TRANSPORT EQUATION USING FINITE DIFFERENCES ##########################


### IMPORTATION
import numpy as np
import math
import sys
from matplotlib import pyplot as plt
from matplotlib import animation

#################################################################################################

### DEFINITION ELLIPTIC FUNCTION J
def FunctionQuad(NT,NX,T,L,alpha,state_apriori,gamma,mu,zeta):
    ### OBSERVER - MOMENT OPERATOR
    # First moment
    operator_moment_1 = (L/NX)*np.linspace(0, L, NX+1)
    operator_moment_1[NX]=0
    # Second moment
    operator_moment_2= (L/NX)*np.square(np.linspace(0, L, NX+1))
    operator_moment_2[NX]= 0
    # Concatenate
    observer = np.vstack((operator_moment_1, operator_moment_2))
    observer_transpose = np.transpose(observer)

    ### DYNAMIC FLOW PHI
    CFL = b*(T/NT)*(NX/L)
    flow = np.eye(NX+1,NX+1) + CFL*(-np.eye(NX+1,NX+1) + np.diag(np.ones(NX),1))
    flow_transpose = np.eye(NX+1,NX+1) + CFL*(-np.eye(NX+1,NX+1) + np.diag(np.ones(NX),-1))

    ### NORM SPACE
     # Construction of the norm of the two spaces
    standart_deviation = 0.001
    norm_observation = (T/NT)*(1/standart_deviation)**2*np.eye(2)
    norm_state = (L/NX)*np.eye(NX+1)

    # Computing A size NX+1,NX+1
    A_function = alpha*norm_state
    A_intermediate = gamma*np.transpose(observer).dot(norm_observation).dot(observer)
    A_function += A_intermediate
    for i in range(1,NT+1):
        A_intermediate = flow_transpose.dot(A_intermediate).dot(flow)
        A_function += A_intermediate

    # Computing b size NX+1,1
    state_ap = np.zeros((NX+1,NT+1))
    state_ap[:,0] = state_apriori.copy()
    b_function = gamma*observer_transpose.dot(norm_observation)\
    .dot(mu[:,0]-observer.dot(state_ap[:,0]))
    for j in range(1,NT+1):
        state_ap[:,j]=flow.dot(state_ap[:,j-1])
        b_intermediate = gamma*observer_transpose.dot(norm_observation)\
        .dot(mu[:,j]-observer.dot(state_ap[:,j]))
        for k in range(1,j+1):
            b_intermediate = flow_transpose.dot(b_intermediate)
        b_function += b_intermediate
    
    # Computimg J = 1/2<Azeta,zeta> - <b,zeta> size 1,1
    J = (1/2)*np.transpose(zeta).dot(A_function).dot(zeta) - np.transpose(b_function).dot(zeta)

    return J


### GRADIENT DIRECT COMPUTATION
def GradientFonctionQuad_1(NT,NX,T,L,alpha,state_apriori,gamma,mu,zeta):

    ### OBSERVER - MOMENT OPERATOR
    # First moment
    operator_moment_1 = (L/NX)*np.linspace(0, L, NX+1)
    operator_moment_1[NX]=0
    # Second moment
    operator_moment_2= (L/NX)*np.square(np.linspace(0, L, NX+1))
    operator_moment_2[NX]= 0
    # Concatenate
    observer = np.vstack((operator_moment_1, operator_moment_2))
    observer_transpose = np.transpose(observer)

    ### DYNAMIC FLOW PHI
    CFL = b*(T/NT)*(NX/L)
    flow = np.eye(NX+1,NX+1) + CFL*(-np.eye(NX+1,NX+1) + np.diag(np.ones(NX),1))
    flow_transpose = np.eye(NX+1,NX+1) + CFL*(-np.eye(NX+1,NX+1) + np.diag(np.ones(NX),-1))

    ### NORM SPACE
     # Construction of the norm of the two spaces
    standart_deviation = 0.001
    norm_observation = (T/NT)*(1/standart_deviation)**2*np.eye(2)
    norm_state = (L/NX)*np.eye(NX+1)
 
    ### COMPUTATION OF A
    # Initialisation of the quadratic function
    A_dJ = alpha*norm_state 
    A_intermediate_1 = gamma*observer_transpose.dot(norm_observation).dot(observer)
    A_dJ += A_intermediate_1

    for j in range(1,NT+1):
        A_intermediate_1 = flow_transpose.dot(A_intermediate_1).dot(flow)
        A_dJ += A_intermediate_1
        
    ### COMPUTATION OF b
    state_a = np.zeros((NX+1,NT+1)) # Saving the solution of the transport equation
    state_a[:,0] = state_apriori.copy() # Initial condition set as a gaussian
    b_dJ = gamma*observer_transpose.dot(norm_observation)\
    .dot(mu[:,0] - observer.dot(state_a[:,0]))
    # Computation of the state time dependant with a priori as initial condition

    for i in range(1,NT+1):
        # Computation of the solution knowing a priori
        state_a[:,i] = np.dot(flow,state_a[:,i-1])
        # Gap between observation and modelisation
        b_intermediate_1 = gamma*np.transpose(observer).dot(norm_observation)\
        .dot(mu[:,i] - observer.dot(state_a[:,i]))
        for k in range(1,i+1):
            b_intermediate_1= flow_transpose.dot(b_intermediate_1)
        b_dJ += b_intermediate_1
                  
    dJ = A_dJ.dot(zeta) - b_dJ

    return dJ


### GRADIENT 4DVAR
def GradientFonctionQuad_2(NT,NX,T,L,alpha,state_apriori,gamma,mu,zeta):

    ### OBSERVER - MOMENT OPERATOR
    # First moment
    operator_moment_1 = (L/NX)*np.linspace(0, L, NX+1)
    operator_moment_1[NX]=0
    # Second moment
    operator_moment_2= (L/NX)*np.square(np.linspace(0, L, NX+1))
    operator_moment_2[NX]= 0
    # Concatenate
    observer = np.vstack((operator_moment_1, operator_moment_2))
    observer_transpose = np.transpose(observer)

    ### DYNAMIC FLOW PHI
    CFL = b*(T/NT)*(NX/L)
    flow = np.eye(NX+1,NX+1) + CFL*(-np.eye(NX+1,NX+1) + np.diag(np.ones(NX),1))
    flow_transpose = np.eye(NX+1,NX+1) + CFL*(-np.eye(NX+1,NX+1) + np.diag(np.ones(NX),-1))

    ### NORM SPACE
    # Construction of the norm of the two spaces
    standart_deviation = 0.001
    norm_observation = (T/NT)*(1/standart_deviation)**2*np.eye(2)
    inv_norm_observation = (standart_deviation)**2/(T/NT)*np.eye(2)
    norm_state = (L/NX)*np.eye(NX+1)
    inv_norm_state = (NX/L)*np.eye(NX+1)

    ### INITIALISATION STATE VECTORS
    # Adjoint state 
    adjoint_state = np.zeros((NX+1,NT+1))
    # State
    state_4dvar = np.zeros((NX+1,NT+1)) # current state
    state_4dvar[:,0] = state_apriori.copy() + zeta

    # Comnputation of solution knowing the initial condition with k observation
    for i in range(0,NT):
        state_4dvar[:,i+1] = np.dot(flow,state_4dvar[:,i])
        
    # Computation of the adjoint state
    adjoint_state[:,NT] = np.zeros(NX+1)
    for ii in range(NT,0,-1):
        adjoint_state[:,ii-1] = np.dot(flow_transpose, adjoint_state[:,ii])\
        + observer_transpose.dot(norm_observation)\
        .dot(mu[:,ii-1]-np.dot(observer,state_4dvar[:,ii-1]))

    # Computation of the gradient
    grad_J = alpha*norm_state.dot(zeta) - gamma*adjoint_state[:,0]
    
    return grad_J


###############################################################################################


### PHYSICAL PARAMETERS
b = 0.2  # Depolymerisation speed
L = 1.0  # Domain size
T = 5.0  # Integration time

### NUMERICAL PARAMETERS
NT = 50 # Number of time steps
NX = 50 # Initial number of grid points
step_save = 10 # Numbers of saving (to plot the solution)


### OBSERVER - MOMENT OPERATOR
# First moment
operator_moment_1 = (L/NX)*np.linspace(0, L, NX+1)
operator_moment_1[NX]=0
# Second moment
operator_moment_2= (L/NX)*np.square(np.linspace(0, L, NX+1))
operator_moment_2[NX]= 0
# Concatenate
observer = np.vstack((operator_moment_1, operator_moment_2))

### DYNAMIC FLOW PHI
CFL = b*(T/NT)*(NX/L)
flow = np.eye(NX+1,NX+1) + CFL*(-np.eye(NX+1,NX+1) + np.diag(np.ones(NX),1))
flow_transpose = np.eye(NX+1,NX+1) + CFL*(-np.eye(NX+1,NX+1) + np.diag(np.ones(NX),-1))

### NORM SPACE
# Construction of the norm of the two spaces
standart_deviation = 0.001
norm_observation = (T/NT)*(1/standart_deviation)**2*np.eye(2)
inv_norm_observation = (standart_deviation)**2/(T/NT)*np.eye(2)
norm_state = (L/NX)*np.eye(NX+1)
inv_norm_state = (NX/L)*np.eye(NX+1)


### DATA GENERATION ############################################################################

### INITIAL CONDITION
x = np.linspace(0.0,L,NX+1)
sigma = L/10
gaussienne = (sigma*np.sqrt(2*np.pi))**-1*np.exp(-(x-L/2)**2/(2*sigma**2)) # Gaussienne
gaussienne[NX] = 0

### VECTOR INITIALISATION
state = np.zeros((NX+1,NT+1)) # Saving the solution of the transport equation
state[:,0] = gaussienne.copy() # Initial condition set as a gaussian
step_register = int((NT+1)/step_save)
mu = np.zeros((2,NT+1)) # First and second order moment

### DIRECT COMPUTATION OF THE SOLUTION
for i in range(0,NT):
    # Calculation of the moment
    mu[:,i] = np.dot(observer,state[:,i])
    # Dynamic computation
    state[:,i+1] = np.dot(flow,state[:,i]) 


###############################################################################################

# State Initialisation
gaussienne_d = (sigma*np.sqrt(2*np.pi))**-1*np.exp(-(x-L/3)**2/(2*sigma**2))
gaussienne_d[NX]=0
double_ic = np.cos(4*np.pi*x/L+np.pi)+ np.ones(NX+1)
# Initial state a priori
state_apriori = gaussienne_d.copy()
# Weight on the measure
gamma = 1
# Weight on the a priori
alpha = 1



###############################################################################################
### TEST  #####################################################################################

npoint = 50
zeta = np.random.rand(NX+1)
beta_liste = np.exp(np.linspace(-1.0,-15.0,npoint))
print(beta_liste)
n_zeta = np.transpose(zeta).dot(norm_state).dot(zeta)
gradJ_0_array = np.zeros(npoint)
n_vecteur1_array = np.zeros(npoint)
n_vecteur2_array = np.zeros(npoint)
epsilon = np.zeros(npoint)

for iii, beta in enumerate(beta_liste):
    delta_zeta = beta*np.random.rand(NX+1)

    gradJ_0 = FunctionQuad(NT,NX,T,L,alpha,state_apriori,gamma,mu,zeta+beta*delta_zeta)\
    -FunctionQuad(NT,NX,T,L,alpha,state_apriori,gamma,mu,zeta)
    gradJ_0_array[iii] = gradJ_0/beta

    vecteur2 = GradientFonctionQuad_2(NT,NX,T,L,alpha,state_apriori,gamma,mu,zeta)
    n_vecteur2_array[iii] = vecteur2.transpose().dot(delta_zeta)

    epsilon[iii] = abs(gradJ_0_array[iii]-n_vecteur2_array[iii])

### PLOTS !!

plt.loglog()
plt.xlabel('beta')
plt.ylabel('norme')
#plit.ylabel("difference")
p0 = plt.plot(beta_liste,gradJ_0_array, label ="difference")
p1 = plt.plot(beta_liste,n_vecteur2_array, label = "adjoint")
# p0 = plt.plot(beta_liste, epsilon, label = "écart gradient")
plt.title("Comparaison de calculs gradient de J")
plt.show()