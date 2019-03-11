## IMPORTATION
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import sys
# optimization function
from scipy.optimize import minimize
# dense to sparse
from scipy.sparse import csr_matrix



### PHYSICAL PARAMETERS
b = 3.0  # Depolymerisation speed
a = 1.0
c0 = 0.0
L = 20.0  # Domain size
T = 10.0  # Integration time

### NUMERICAL PARAMETERS
NT = 150  # Number of time steps
NX = 100   # Initial number of grid points

### INITIAL CONDITION
x = np.linspace(0.0,L,NX)
cmax = 0.01 # Maximum concentration of polymers
imax = L/2
sigma = 2
def Gaussienne(x,c,i,s):
    gaussienne = c*np.exp(-(x-i)**2/(2*s**2)) # Gaussienne
    Ncrop = math.ceil(NX*0.1)
    return gaussienne

fig1, ax1 = plt.subplots()
ax1.plot(np.linspace(0,L,NX), Gaussienne(x,cmax,imax,sigma), 'r--', label='initial')
ax1.plot(np.linspace(0,L,NX), Gaussienne(x,cmax,7,sigma), 'g--', label='t1')
ax1.plot(np.linspace(0,L,NX), Gaussienne(x,cmax,4,sigma), 'g--', label='t2')
ax1.plot(np.linspace(0,L,NX), Gaussienne(x,cmax,2,sigma), 'g--', label='t3')
legend = ax1.legend(loc='upper right', shadow=True, fontsize='x-large')
ax1.set(xlabel='size (mm)', ylabel='concentration',
       title='Transport equation')
plt.show()
