import numpy as np
import sys
import random
# plot
import matplotlib.pyplot as plt
from matplotlib import animation, rc
#
from DeepNN.network import Network
from DeepNN.fclayer import FCLayer
from DeepNN.activation import ActivationLayer
from DeepNN.activations import tanh, tanh_prime
from DeepNN.losses import mse, mse_prime

# Physical data
l = 10
tau = 5
dep = 2
# Numerical data
nx = 10
dx = l/nx
nt = 10
dt = tau/nt
x_grid = np.linspace(0,l,nx)
# forward_operator = l/nx*np.tri(nt, nx, 0, dtype=int)
forward_operator = 0.1*np.eye(nt)

# training data
ntrain = 100
ndata = nt
st = np.zeros((ntrain,1,nx))
x_train = np.zeros((ntrain,1,nt))
y_train = np.zeros((ntrain,1,ndata))
for i in range(0,ntrain):
    # value for x
    mu = random.uniform(1,l-1)
    sigma = random.uniform(0.1,0.5)
    st[i,0,:] = (sigma*np.sqrt(2*np.pi))**-1*np.exp(-(x_grid-mu)**2/2*sigma**2)
    x_train[i,0,:] = forward_operator.dot(st[i,0,:]) 
    y_train[i,0,:] = np.interp(np.linspace(0,l,ndata), x_grid, st[i,0,:])

# network
net = Network()
# number of layers
nlayers = 10
nx1 = np.linspace(nt,ndata,nlayers+1)
nx1 = nx1.astype(int)
i=0
while i < nlayers:
    net.add(FCLayer(nx1[i], nx1[i+1]))
    net.add(ActivationLayer(tanh, tanh_prime))
    i+=1
# train
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=1000, learning_rate=0.01)

# test
out = net.predict(x_train)

# visualize
fig, ax = plt.subplots()
for i in range(10):
    ax.plot(np.linspace(0,l,ndata), out[i][0,:], label= "s{}".format(i))
legend = ax.legend(loc='upper left', shadow=True, fontsize='x-large')
legend.get_frame().set_facecolor('C0')
ax.set(xlabel='labelx', ylabel='labely',
       title='NN reconstruction')
plt.show()
