import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import *
from math import *

# Constants and parameters

w = 9 # Frequency for x(t) and y(t)
Omega0 = 9 #eB0
Omega1 = 1 #eB1 #1
Delta = w-Omega0
Omega = np.sqrt(Delta**2 + Omega1**2)
w1 = Omega/Omega1
w2 = Delta/Omega1
T = 2*pi/Omega
t = np.linspace(0, 5*T, 10000)
# a and b
aa = ((w1+w2)/(2*w1))*np.exp((1j*Omega*t)/2)+((w1-w2)/(2*w1))*np.exp(-1j*Omega*t/2)
bb = -(w1-w2)*((w1+w2)/(2*w1))*np.exp((1j*Omega*t)/2)+(w1+w2)*((w1-w2)/(2*w1))*np.exp(-1j*Omega*t/2)
a = aa*np.exp(-1j*w*t/2)
b = bb*np.exp(1j*w*t/2)

#first probability
plt.subplot(311)
plt.grid()
plt.ylim(0,1.1)
plt.plot(t, a*np.conj(a)+b*np.conj(b), label = '<Ψ|Ψ>')
plt.xlabel('t')
plt.legend(loc='lower right')

plt.subplot(312)
plt.grid()
plt.ylim(0,1.1)
plt.plot(t, a*np.conj(a), label = '|<↑|Ψ>|²')
plt.xlabel('t')
plt.legend(loc='lower right')

plt.subplot(313)
plt.grid()
plt.ylim(0,1.1)
plt.plot(t, b*np.conj(b), label = '|<↓|Ψ>|²')
plt.xlabel('t')
plt.legend(loc='lower right')

# show the plot
plt.show()