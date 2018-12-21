# -*- coding : utf-8 -*-

## COMPUTATION OF CONJUGATE GRADIENT USING PYTHON LIBRARY
## FOR THE MINIMISATION OF J = 1/2XAX-bX

###################### 1D TRANSPORT EQUATION USING FINITE DIFFERENCES ##########################

### IMPORTATION
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import math
import sys

### PARAMETERs ############################################################################

class Foo():

    def __init__(self):
        ### PHYSICAL PARAMETERS
        b = 0.2  # Depolymerisation speed
        L = 1.0  # Domain size
        T = 5.0  # Integration time

        ### NUMERICAL PARAMETERS
        self.NT = 50  # Number of time steps
        self.NX = 50   # Initial number of grid points

        ### OBSERVER - MOMENT OPERATOR
        # First moment
        operator_moment_1= L/self.NX*np.linspace(0, L, self.NX)
        # Second moment
        operator_moment_2= L/self.NX*np.square(np.linspace(0, L, self.NX))
        # Concatenate
        self.observer = np.vstack((operator_moment_1, operator_moment_2))
        self.observer_transpose = np.transpose(self.observer)

        ### DYNAMIC FLOW PHI
        CFL = b*(T/self.NT)*(self.NX/L)
        self.flow = np.eye(self.NX,self.NX) + CFL*(-np.eye(self.NX,self.NX) + np.diag(np.ones(self.NX-1),1))
        self.flow_transpose = np.eye(self.NX,self.NX) + CFL*(-np.eye(self.NX,self.NX) + np.diag(np.ones(self.NX-1),-1))

        ### INITIAL CONDITION
        self.x = np.linspace(0.0,L,self.NX+1)
        sigma = L/10
        self.gaussienne = (sigma*np.sqrt(2*np.pi))**-1*np.exp(-(self.x-L/2)**2/(2*sigma**2)) # Gaussienne
        self.gaussienne[self.NX] = 0

        ### GRADIENT PARAMETER
        self.iteration_max = 100
        self.gamma = 10000000
        self.alpha = 1

        ### NORM
        # Construction of the norm of the two spaces
        self.norm_observation = T/self.NT*np.eye(2)
        self.inv_norm_observation = 1/(T/self.NT)*np.eye(2)
        self.norm_state = (L/self.NX)*np.eye(self.NX)
        self.inv_norm_state = (self.NX/L)*np.eye(self.NX)

        

### DATA GENERATION ############################################################################

def DataGeneration():

    ### DOWNLOADING PARAMETERS
    foo = Foo()

    ### VECTOR INITIALISATION
    state = np.zeros((foo.NX,foo.NT+1)) # Saving the solution of the transport equation
    state[:,0] = foo.gaussienne[:-1].copy()
    mu = np.zeros((2,foo.NT+1)) # First and second order moment

    ### DIRECT COMPUTATION OF THE SOLUTION
    for i in range(0,foo.NT):
        # Calculation of the moment
        mu[:,i] = np.dot(foo.observer,state[:,i])
        # Dynamic computation
        state[:,i+1] = np.dot(foo.flow,state[:,i]) 

    # Return observation vector
    return mu

### CRITERION COMPUTATION ###########################################################################

def CriterionJ(zeta):

    ### DOWNLOADING PARAMETERS
    foo = Foo()

    ### DOWNLOADING OBSERVATIONS
    mu = DataGeneration()

    ### MATRIX COMPUTATION
    # Computation of the Hessian A
    A_quadratic = foo.alpha*foo.norm_state.copy()
    A_inter = foo.gamma*foo.observer_transpose.dot(foo.norm_observation).dot(foo.observer)
    A_quadratic += A_inter
    for i in range(1,foo.NT+1):
        A_inter = foo.flow_transpose.dot(A_inter).dot(foo.flow)
        A_quadratic += A_inter

    # Computation of b
    # No a priori on the initial solution (equal to zero)
    state_apriori = np.zeros((foo.NX,foo.NT+1))
    b_linear = foo.gamma*foo.observer_transpose.dot(foo.norm_observation)\
    .dot(mu[:,0]- foo.observer.dot(state_apriori[:,0]))
    for j in range(1,foo.NT+1):
        state_apriori[:,j] = foo.flow.dot(state_apriori[:,j-1])
        b_inter = foo.gamma*foo.observer_transpose.dot(foo.norm_observation)\
        .dot(mu[:,j]-foo.observer.dot(state_apriori[:,j]))
        for k in range(1,j+1):
            b_inter = foo.flow_transpose.dot(b_inter)
        b_linear += b_inter

    ### CRITERION COMPUTATION
    # Computimg J = 1/2<Azeta,zeta> - <b,zeta> size 1,1
    J = (1/2)*np.transpose(zeta).dot(A_quadratic).dot(zeta) - np.transpose(b_linear).dot(zeta)

    return J

### GRADIENT COMPUTATION ############################################################################

def CriterionGrad(zeta):

    ### DOWNLOADING PARAMETERS
    foo = Foo()

    ### DOWNLOADING OBSERVATIONS
    mu = DataGeneration()

    ### MATRIX COMPUTATION
    # Computation of the Hessian A
    A_quadratic = foo.alpha*foo.norm_state.copy()
    A_inter = foo.gamma*foo.observer_transpose.dot(foo.norm_observation).dot(foo.observer)
    A_quadratic += A_inter
    for i in range(1,foo.NT+1):
        A_inter = foo.flow_transpose.dot(A_inter).dot(foo.flow)
        A_quadratic += A_inter

    # Computation of b
    # No a priori on the initial solution (equal to zero)
    state_apriori = np.zeros((foo.NX,foo.NT+1))
    b_linear = foo.gamma*foo.observer_transpose.dot(foo.norm_observation)\
    .dot(mu[:,0]- foo.observer.dot(state_apriori[:,0]))
    for j in range(1,foo.NT+1):
        state_apriori[:,j] = foo.flow.dot(state_apriori[:,j-1])
        b_inter = foo.gamma*foo.observer_transpose.dot(foo.norm_observation)\
        .dot(mu[:,j]-foo.observer.dot(state_apriori[:,j]))
        for k in range(1,j+1):
            b_inter = foo.flow_transpose.dot(b_inter)
        b_linear += b_inter

    ### CRITERION COMPUTATION
    # Computimg GradJ = Azeta - b size 1,NX-1
    GradJ = A_quadratic.dot(zeta) - b_linear

    return GradJ

### HESSIAN COMPUTATION #############################################################################

def CriterionHess(zeta):

    ### DOWNLOADING PARAMETERS
    foo = Foo()

    ### MATRIX COMPUTATION
    # Computation of the Hessian A
    A_quadratic = foo.alpha*foo.norm_state.copy()
    A_inter = foo.gamma*foo.observer_transpose.dot(foo.norm_observation).dot(foo.observer)
    A_quadratic += A_inter
    for i in range(1,foo.NT+1):
        A_inter = foo.flow_transpose.dot(A_inter).dot(foo.flow)
        A_quadratic += A_inter

    return A_quadratic

### NEWTON GRADIENT ALGORITHM #######################################################################
foo = Foo()
zeta0 = np.zeros(foo.NX)
res = minimize(CriterionJ, zeta0, method='BFGS',\
               jac=CriterionGrad, \
               options = {'disp': True} )

plt.xlabel(u'$x$', fontsize=5)
plt.ylabel(u'$y$', fontsize=5, rotation=0)
plt.title("alpha/gamma = {0}, iteration = {1}"\
.format(foo.alpha/foo.gamma, res.nit))
p0 = plt.plot(foo.x, foo.gaussienne, label= "condition initiale")
p1 = plt.plot(foo.x[:-1], res.x, label= "estimateur 4dvar")
p2 = plt.plot(foo.x, np.zeros(foo.NX+1), label= "a priori")
plt.legend()
plt.show()
