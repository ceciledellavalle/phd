import numpy as np
import sys

from DeepNN.network import Network
from DeepNN.fclayer import FCLayer
from DeepNN.activation import ActivationLayer
from DeepNN.activations import tanh, tanh_prime
from DeepNN.losses import mse, mse_prime

# training data
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
print(x_train.shape)
sys.exit()
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# network
net = Network()
net.add(FCLayer(2, 3))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(3, 1))
net.add(ActivationLayer(tanh, tanh_prime))

# train
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

# test
out = net.predict(x_train)
