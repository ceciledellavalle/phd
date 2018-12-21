# -*- coding : utf-8 -*-

### IMPORTATION PYTHON
import numpy as np
import math
import sys

### IMPORTATION OF MY FUNCTION
from offlinee import OfflineError
from offlinee import Distance
from offlinee import Activation
from recurdelta import ComputeDelta



### CONSTRUCTION OF A NEURAL NETWORK

class NeuralNetworkStructure:
    """
    Neural network is defined by :
    nl -- int - number of layers ;
    nls -- row array - number of neuron in each layer (input = 0, output = nl-1) ;
    then it generates automatically the following arguments :
    card -- int - total number of neurons ;
    wlI -- row array - weight in each layer (input = 0, output = cardinal N -1) ;
    nwlI -- int - total number of connection ;
    thetals -- list of array - activation rate of each neuron in each layer.
    """

    def __init__(self,nl,nls):
        self.nl = nl
        self.nls = nls
        self.card = np.sum(nls)
        # Computing the number of connection
        self.nwlI = np.sum(np.multiply(nls[:-1],nls[1:]))
        # Generate a row array of zero
        self.wlI = np.zeros(self.nwlI)
        # Generate a row array of 1 for theta
        # first layer neuron doesn't have theta
        self.thetals = np.zeros(self.card-nls[0])


    ### COMPUTATIOM OF NEURAL NETWORK
    def Predict(self,pattern_row):
        """
        Evaluate output vector yi corresponding to input pattern pi
        input is a numerical vector of size corresponding to neural network nls[-1]
        """
        nls = self.nls
        sl = np.cumsum(nls)
        # Test des données d'entrée
        if nls[0] != np.size(pattern_row):
            raise TypeError("Les données d'entrée n'ont pas la bonne dimension.")
        # Initialisation of neurons input
        y = np.zeros(self.card)
        y[0:nls[0]] = pattern_row
        # Indexation to skim through :
        index_w = 0 # the weight vector
        index_l = 0 # the neuron in layers
        # Computation in each layer
        for i in range(0,self.nl-1):
            # Propagation function for layer i
            ui = np.dot(y[index_l:index_l+nls[i]],\
            self.wlI[index_w:index_w+nls[i]*nls[i+1]]\
            .reshape(nls[i],nls[i+1]))
            index_w += nls[i]*nls[i+1]
            # Activation function for layer i
            ui -= self.thetals[index_l:index_l+nls[i+1]]
            index_l += nls[i]
            ai = Activation(ui)
            # Output function for layer i
            y[sl[i]:sl[i+1]] = ai
        # Return result of all layer
        return y


    ### LEVENBERG-MARQUARDT ITERATION
    # An iteration of the Levenberg-Marquardt training algorithm
    def LMiteration(self,pattern,output,error,mu):
        """ trainingset -- array of two row (pattern, output) ;
        error -- error initialisation ;
        mu -- number - balance parameter ;
        Computes an iteration of the Levenberg-Marquardt training algorithm
        """
        beta = 10
        # Definition of the number mo of output cells
        mo = self.nls[-1]
        # Initialisation of the vector error of size mo
        epi = np.zeros(mo)
        error0 = error
        JpJp = np.zeros((self.nwlI,self.nwlI))
        Jpep = np.zeros(self.nwlI)

        # Indexation to skim through the training pattern
        line = 0
        for row in pattern:
            # Evaluation the output vector y corresponding to the pattern
            y = self.Predict(row)
            epi = np.square(y[self.card-1-mo:self.card-1] - output[line,:])

            # Initialisation of the Jacobian matrix
            Jp = np.zeros((mo,self.nwlI))

            # Incrementing neuron numerotation
            sl = np.concatenate(([0],np.cumsum(self.nls)))
            wsl = np.cumprod(self.nls)

            # Incrementing the number of layers
            # Starting from output layer
            index_l = self.nl-1


            # For each output cell
            # backward filling
            # from mo number of output to 0
            for h in range(0,mo):
                # Initialisation of index k
                # Incrementing the number of connection
                k = self.nwlI-1
                # For each neuron j in layers
                # Except input layers that is not needed
                for j in range(self.card-1,sl[1]-1,-1):
                    # Conpute the index layer of j
                    index_l = np.min(np.where(sl>j))
                    for i in range(sl[index_l-1],sl[index_l-2],-1):
                        Jp[h,k] = ComputeDelta(self,h,j,index_l,sl,wsl,y)*y[i]
                        k-=1
            # Incrementing line to skim through output
            line +=1
            # Computing hessian matrix and gradient
            JpJp += np.transpose(Jp).dot(Jp)
            Jpep += np.transpose(Jp).dot(epi)
        # Assemble and solve the gradient gradw
        interim_matrix = np.linalg.inv(JpJp + mu*np.eye(self.nwlI))
        gradw = -interim_matrix.dot(Jpep)

        # Move the matrix weight
        # according to the computed gradient
        self.wlI = self.wlI + gradw

        error1 = OfflineError(self,pattern,output)
        # Test and replace the balance between the gradient method/newthod method
        if error1 < error0:
            mu = mu/beta
        else:
            mu = mu*beta

        error0 = error1
        return error0,mu
