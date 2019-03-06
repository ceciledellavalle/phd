# -*- coding : utf-8 -*-

#########################     PLOTTING SOLUTION OF 1D TRANSPORT EQUATION   #########################

### IMPORTATION PYTHON
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

### FUNCTION DEFINITION
def PlotDymamicSolution(xmax,ymax,vectorX,vectorY,nbData,vectorT):
    """ Function that allows to plot animate with a constant absisse vectorX (shape Nx1)
    and a solution evolving through time vectorY (shape NxnbData)."""

    # Set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = plt.axes(xlim=(0, xmax), ylim=(0, ymax))
    line, = ax.plot([], [], lw=2)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)


    # Initialization function: plot the background of each frame
    def Init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text

    # Animation function.  This is called sequentially
    def Animate1DT(i):
        x = vectorX
        y = vectorY[:,i]
        line.set_data(x, y)
        time_text.set_text('time = %.1f'%vectorT[i])
        return line, time_text

    anim = animation.FuncAnimation(fig, Animate1DT, \
    init_func=Init,\
    frames=nbData, interval=50, blit=True)

    return anim
