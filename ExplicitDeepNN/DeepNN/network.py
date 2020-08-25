import sys

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
    def predict(self, input_data, obs_data):
        # sample dimension first
        samples = len(input_data)
        result = []
        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            outputk = input_data[i]
            obs = obs_data[i]
            for layer in self.layers:
                layer.initial_point(outputk,obs)
                output = layer.forward_propagation(output)
                outputk = layer.result_onestep_point(outputk)
            result.append(output)
        return result

    # train the network
    def fit(self, x_train, y_train, obs_data, epochs, learning_rate):
        # sample dimension first
        samples = len(x_train)
        # training loop
        i = 0
        while i < epochs:
            err = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                outputk = x_train[j]
                obs = obs_data[j]
                for layer in self.layers:
                    layer.initial_point(outputk,obs)
                    output = layer.forward_propagation(output)
                    outputk = layer.result_onestep_point(outputk)
                # compute loss (for display purpose only)
                err += self.loss(y_train[j], output)
                # backward propagation
                error = self.loss_prime(y_train[j], output)
                errork = 0*error
                for layer in reversed(self.layers):
                    layer.initial_point(errork,0)
                    error = layer.backward_propagation(error, learning_rate)
                    errork = layer.result_onestep_point(error)
            # calculate average error on all samples
            err /= samples
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))
            i+=1