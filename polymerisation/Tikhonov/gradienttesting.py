# -*- coding : utf-8 -*-

## TIKHONOV REGULATION


### IMPORTATION
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import sys


##############################################################################
### GRADIENT TESTING #######)##################################################
#Initialisation
# npoint = 50
# beta_liste = np.exp(np.linspace(0.0,-20.0,npoint))
# gradJ0 = np.zeros(npoint)
# gradJ1 = np.zeros(npoint)
# epsilon = np.zeros(npoint)
# zeta = np.random.rand(NX)
# #For decreasing beta
# for j, beta in enumerate(beta_liste):
#     zeta_1 = np.random.rand(NX)
#     norm_zeta_1 = (L/NX)*np.linalg.norm(zeta_1)
#     zeta_1 = zeta_1/norm_zeta_1
#     # Compute difference
#     gradJ0[j] = Rosen(zeta+beta*zeta_1,\
#     phi.T,T,NT,L,NX,moment0)\
#     - Rosen(zeta,\
#     phi.T,T,NT,L,NX,moment0)
#     gradJ0[j] = gradJ0[j]/beta
#     # Compute grad
#     gradJ1v = Rosen_der(zeta,\
#     phi.T,T,NT,L,NX,moment0)
#     gradJ1[j] = gradJ1v.dot(zeta_1)
#     # Compute the error between difference and grad
#     epsilon[j] = abs(gradJ0[j]-gradJ1[j])

### PLOTS !!
# plt.loglog()
# plt.xlabel('beta')
# plt.plot(beta_liste,gradJ0, label ="difference")
# plt.plot(beta_liste,gradJ1, label = "gradient")
# plt.legend()

##############################################################################
##############################################################################

### CONTROL OF ROSEN SOLUTION
# Create plots with pre-defined labels.
# moment0_sim = matrix_psi.dot(state_init)
# fig1, ax1 = plt.subplots()
# ax1.plot(np.linspace(0,T,NT), moment0, 'b--', label='C state')
# ax1.plot(np.linspace(0,T,NT), moment0_sim, 'b-', label='Psi y')
# legend = ax1.legend(loc='lower right', shadow=True, fontsize='x-large')
# ax1.set(xlabel='time (s)', ylabel='moment',\
#          title='Evolution of the 0 order moment')
