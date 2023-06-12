# -*- coding: utf-8 -*-
"""
Animation of probability density of a particle in an infinite square well

Created on Tue Mar 29 21:35:14 2022

@author: Alexandre El-Masry
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from matplotlib import animation


pi = np.pi
L = 1  # length of box set to 1 for visual simplicity
Nx = 350  # number of divisions set to 350 because it is the ideal number before wavefunction because meaningless
xvals = np.linspace(0, L, Nx)
tvals = np.linspace(0, 100, Nx)
A = np.sqrt(2/L)
c = np.empty(Nx)  # linear combination coefficients
# hbar set to 10e-3 because the wavefunction becomes meangless when smaller,
hbar = 1.05457182E-3
# this is one of the flaws mentioned in the paper.
E = np.empty(Nx)  # Energy states
m = 1  # mass set to 1 for simplicity

# animation plot configuration
fig = plt.figure()
ax = plt.axes(xlim=(0, 1), ylim=(0, 5))
line, = ax.plot([], [], lw=2)
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

# time independent wavefunction solution to infinite square well (small psi)


def psi(x, Nx):
    wf = A*np.sin(Nx*pi*x/L)
    return wf


# loop to compute the sum of all linear coefficients and energy states
for i in range(Nx):
    c[i] = A*integrate.quad(psi, 0, L, args=(i,))[0]
    E[i] = (i**2*pi**2*hbar**2/(2*m*L**2))


def init():
    line.set_data([], [])
    return line,

# animation body


def animate(tvals):
    x = np.linspace(0, L, Nx)
    y = sum(c[i]*(A*np.sin(i*pi*xvals/L))*np.exp(-1j*E[i]*tvals/hbar)
            for i in range(1, Nx))
    f = np.abs(y**2)
    line.set_data(x, f)
    return line,


# plotting fucntion calls
anim = animation.FuncAnimation(
    fig, animate, init_func=init, frames=20000, interval=150, blit=False)
plt.title("Time-dependent Wavefunction of a particle in a box")
plt.ylabel("|\u03C8(x,t)^2|")
plt.xlabel("x")
plt.grid()
plt.show()
