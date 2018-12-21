# -*- coding : utf-8 -*-
# Procedure to compute the accumulated error.

### IMPORTATION PYTHON
import numpy as np
import math
import sys

def OfflineError(ann,pattern,output):
    """ Offline error is a procedure to compute the accumulated error
    Given a neural network (neurons, weight) and a training set (pattern, output)
    It compute the cumulated distance petween the actual result and the expected output
    neuralnetwor -- NeuralNetworkStructure
    trainingset -- list of two array, the input and the output
    """
    E = 0
    line = 0
    n = ann.card-ann.nls[-1]
    for row in pattern:
        # Calculate the correspondint y = F(p)
        # Where F is the function of convolution
        #of the neuralm network
        y = ann.Predict(row)
        # Compute the error related to pattern p
        E = E + Distance(y[n:],output[line,:])
        # Incrementing line index
        line +=1
    # Return the computed error
    return E

def Distance(a,b):
    """The function Distance compute the norm l2
    of two vector of the same size nx1
    a -- numpy array size 1xn
    b -- numpy array size 1xn"""
    dist = np.linalg.norm(a-b)
    # Return the distance
    return dist

def Activation(matrix):
    """ the activation function takes an integer as parameters
    and compute an hyperbolic tangent.
    """
    tanh = np.tanh(matrix)
    return tanh

def Activation_dev(matrix):
    """ the activation function takes an integer as parameters
    and compute an hyperbolic tangent.
    """
    tanh_dev = np.ones(np.shape(matrix)) - np.square(np.tanh(matrix))
    return tanh_dev
