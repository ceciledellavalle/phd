import numpy as np
import sys
import random
#
from DeepNN.network import Network
from DeepNN.fclayer import FCLayer
from DeepNN.explicit import ExplicitLayer
from DeepNN.activation import ActivationLayer
from DeepNN.activations import tanh, tanh_prime
from DeepNN.losses import mse, mse_prime
# plot
import matplotlib.pyplot as plt
from matplotlib import animation, rc


# Physical data
l = 10
tau = 5
dep = 2
# Numerical data
nx = 12
dx = l/nx
nt = 10
dt = tau/nt
x_grid = np.linspace(0,l,nx)
forward_operator = l/nx*np.tri(nx, nt, 0, dtype=int)
forward_operator = forward_operator.T

# training data
ndata = 50
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
#    x_train[i,:]= forward_operator.T.dot(obs_data[i,:])

# network
net = Network()
# number of layers
nlayers = 20
i=0
while i < nlayers:
    net.add(FCLayer(nx, nx))
    net.add(ActivationLayer(tanh, tanh_prime))
    net.add(ExplicitLayer(forward_operator,0.01))
    i+=1

# train
net.use(mse, mse_prime)
net.fit(x_train, y_train, obs_data, epochs=1000, learning_rate=0.005)

# test
out = net.predict(x_train,obs_data)

fig, ax = plt.subplots()
ax.plot(x_grid, y_train[0,:],'k-', label= "initial")
ax.plot(x_grid, out[0],'k+-', label= "initial")
ax.plot(x_grid, y_train[1,:],'g-', label= "initial")
ax.plot(x_grid, out[1],'g+-', label= "initial")
legend = ax.legend(loc='upper left', shadow=True, fontsize='x-large')
legend.get_frame().set_facecolor('C0')
ax.set(xlabel='labelx', ylabel='labely',
       title='KALMAN RECONSTRUCTION')

plt.show()