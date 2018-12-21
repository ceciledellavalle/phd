# -*- coding : utf-8 -*-

#########################################################
##
##                 ARTIFICIAL NEURAL NETWORK
##                      FIRST ATTEMPT
##
#########################################################

# Offline supervised learning algorithm;


### IMPORTATION PYTHON
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import sys
import scipy.optimize as opt

### IMPORTATION OF MY FUNCTION
from offlinee import OfflineError
from offlinee import Distance
from anndef import NeuralNetworkStructure

### INITIALISATION
erroropt = 1000000
Kmr = 100
epsilon = 0.01 # maximal expected error on the training

### CREATION OF THE NEURAL NETWORK
nl = 4 # Number of layers
nls = np.array([10,5,5,10]) # Number of cells in each layer
# Test de validité
if nl != np.size(nls):
    raise TypeError("Le nombre de couche est invalide.")
# Initialisation of neural network
nni = NeuralNetworkStructure(nl,nls)

### CREATION OF THE TRAINING SET
# List of tuple (pattern,output)
# From download, etc.
### PATTERN DEFINITION
# Training pattern
ptr = np.linspace(0,1,20)
ttr = 2*np.square(ptr)+ptr
ptr = ptr.reshape(2,10)
# Validation pattern
pva = np.linspace(0,1,10)
tva = np.linspace(0,1,10)
# Testing pattern
pte = np.random.rand(50,3)
tte = np.random.rand(50,2)
Kea = 100 # Admissible iteration if the validation error increases
T = 100 # Number of epoch (training iteration)
# Plot INITIALISATION
plot_error = 10*np.ones(T)



###############################################################################
###############################################################################
###############################################################################

### TRAINING AND VALIDATION
# Multiple restarting method
#for j in range(1,Kmr):
t=0
k=0
mu= 0.01
# Generate a random weight matrix
w0 = np.random.rand(nni.nwlI)
nni.wlI = w0
# Initialization of the error
errortr = OfflineError(nni,ptr,ttr)
errorva0 = errortr

while (t<T)&(errortr>epsilon)&(k<Kea):
    # Training step
    errortr, mu = nni.LMiteration(ptr,ttr,errortr,mu)
    plot_error[t] = errortr

    # Validation step
    errorva1 = OfflineError(nni,pva,tva)
    if errorva1>errorva0:
        k+=1
    else:
        k=0
    t+=1

### TESTING

### PLOTTING
# Data for plotting
tv = np.linspace(0,T,T)

fig, ax = plt.subplots()
ax.plot(tv, plot_error)

ax.set(xlabel='time (t)', ylabel='error (x)',
       title='Error evolving through training')
ax.grid()

#fig.savefig("test.png")
plt.show()
