import numpy as np
from DeepNN.layer import Layer

# inherit from base class Layer
class ExplicitLayer(Layer):

    def __init__(self, operator, rho):
        self.op = operator
        self.rho = rho

    # initialize x_0 and y
    def initial_point(self,x,y):
        self.x = x
        self.y = y

    # returns the gradient descent according to the operator
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.x - self.rho*(np.dot(self.op.T,self.op.dot(self.x) - self.y))-input_data
        self.x = self.output
        return self.output
    
    # increment the value x_k
    def result_onestep_point(self, input_data):
        return self.output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward_propagation(self, output_error, learning_rate):
        input_error = self.x - self.rho*np.dot(self.op.T,self.op.dot(self.x))-output_error
        return input_error