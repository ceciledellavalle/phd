import numpy as np
import sys
import random
# plot
import matplotlib.pyplot as plt
from matplotlib import animation, rc


# Physical data
l = 10
tau = 5
dep = 2
# Numerical data
nx = 120
dx = l/nx
nt = 100
dt = tau/nt
x_grid = np.linspace(0,l,nx)
forward_operator = l/nx*np.tri(nx, nt, 0, dtype=int)
forward_operator = forward_operator.T

# training data
ndata = 20
x_train = np.zeros((ndata,nx))
y_train = np.zeros((ndata,nx))
obs_data = np.zeros((ndata,nt))
print("norm")
for i in range(0,20):
    # value for x
    mu = random.uniform(1,l-1)
    sigma = random.uniform(0.1,0.5)
    y_train[i,:] = (sigma*np.sqrt(2*np.pi))**-1*np.exp(-(x_grid-mu)**2/2*sigma**2)
    obs_data[i,:] = forward_operator.dot(y_train[i,:]) 

# gradient descent
rho = 0.01
nbiter = 20
x = x_train[5,:]
y = obs_data[5,:]
for i in range(nbiter):
    x = x - rho*np.dot(forward_operator.T,forward_operator.dot(x) - y)


# plots
fig, ax = plt.subplots()
ax.plot(x_grid, x,'k-', label= "gd")
ax.plot(x_grid, y_train[5,:],'k+', label= "initial")
legend = ax.legend(loc='upper left', shadow=True, fontsize='x-large')
legend.get_frame().set_facecolor('C0')
ax.set(xlabel='labelx', ylabel='labely',
       title='KALMAN RECONSTRUCTION')

plt.show()