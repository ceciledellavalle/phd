# -*- coding : utf-8 -*-
# Procedure to compute the accumulated error.

### IMPORTATION PYTHON
import numpy as np
import math
import sys


def ComputeDelta(ann,h,j,layer,cumul_nls,wsl,y):
    """Compute the deltahj of neuron j related to the output neuron h."""
    # INITIALISATION of deltahj
    deltahj = 1

    # For the output neuron
    if layer == ann.nl:
        if ann.card-j-1 == h:
            deltahj *= (1-y[ann.card-1-h]**2)
        else :
            deltahj *= 0
    # For the inner layer
    else:
        for s in range(cumul_nls[layer-1],cumul_nls[layer]):
            # Compute the index of the connection between
            # neuron number s and neuron number j
            r = 0
            # number of connection in all previous layers
            r += wsl[layer-2]
            # number of connection in the layer of s before s
            r += (s-cumul_nls[layer-1])*ann.nls[layer-2]
            # connection of neuron s with neuron before j
            r += j-cumul_nls[layer]

            # Call function recursively
            deltahj += (1-y[j]**2)*ann.wlI[r]\
            *ComputeDelta(ann,h,s,layer+1,cumul_nls,wsl,y)

    return deltahj
