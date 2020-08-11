import numpy as np
import sys
import random
# plot
import matplotlib.pyplot as plt

class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []
        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)
        plt.show()
        return result

    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate):
        # sample dimension first
        samples = len(x_train)
        # training loop
        i = 0
        while i < epochs:
            err = 0
            j= random.randint(0,samples-1)
            output = x_train[j]
            # forward propagation
            for layer in self.layers:
                output = layer.forward_propagation(output)

            # compute loss (for display purpose only)
            err += self.loss(y_train[j], output)

            # backward propagation
            error = self.loss_prime(y_train[j], output)
            for layer in reversed(self.layers):
                error = layer.backward_propagation(error, learning_rate)
            # calculate average error on all samples
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))
            i+=1