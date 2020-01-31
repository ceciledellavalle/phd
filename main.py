import numpy as np
import sys
import random
from scipy import signal
#
from DeepNN.network import Network
from DeepNN.fclayer import FCLayer
from DeepNN.explicit import ExplicitLayer
from DeepNN.activation import ActivationLayer
from DeepNN.activations import tanh, tanh_prime
from DeepNN.losses import mse, mse_prime

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
# training data
ndata = 20
x_train = np.zeros((ndata,nx))
y_train = np.zeros((ndata,nt))
for i in range(0,ndata):
    # value for x
    mu = random.uniform(1,l-1)
    sigma = random.uniform(0.1*l,0.9*l)
    epsilon = random.uniform(0,0.1*(sigma*np.sqrt(2*np.pi))**-1)
    x_train[i,:] = (sigma*np.sqrt(2*np.pi))**-1*np.exp((x_grid-mu)**2/(2*sigma**2))
    y_train[i,:] = forward_operator.dot(x_train[i,:]) + epsilon*np.random.rand(nt)

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

# test
out = net.predict(x_train,y_train)

# train
# net.use(mse, mse_prime)
# net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)
