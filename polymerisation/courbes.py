# -*- coding : utf-8 -*-
### Tracer les courbes f_0 et g_0
### Contre exemple du théorème

import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0,1,100)
f0 = 4*np.multiply(1+x,1-x)
g0 = f0 + 20*x**3 - 40*x**2+18*x+2


# Create plots with pre-defined labels.
fig, ax = plt.subplots()
ax = plt.axes(xlim=(0, 1), ylim=(0, 10))
ax.plot(x, f0, 'k--', label='f0=4(1-x)(1+x)')
ax.plot(x, g0, 'k:', label='g0=f0-P2+P3')


legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')

# Put a nicer background color on the legend.
legend.get_frame().set_facecolor('C0')

plt.show()
