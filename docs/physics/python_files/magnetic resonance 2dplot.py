import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import *
from math import *

# Constants and parameters

w = 5 # Frequency for x(t) and y(t)
Omega0 = 5 #eB0
Omega1 = 1
Delta = w-Omega0
Omega = np.sqrt(Delta**2 + Omega1**2)
w1 = Omega/Omega1
w2 = Delta/Omega1
t = np.linspace(0, 100, 10000)  # Time array range 500 is broken into 10000 pieces
# a and b
aa = ((w1+w2)/(2*w1))*np.exp((1j*Omega*t)/2)+((w1-w2)/(2*w1))*np.exp(-1j*Omega*t/2)
bb = -(w1-w2)*((w1+w2)/(2*w1))*np.exp((1j*Omega*t)/2)+(w1+w2)*((w1-w2)/(2*w1))*np.exp(-1j*Omega*t/2)
a = aa*np.exp(-1j*w*t/2)
b = bb*np.exp(1j*w*t/2)


#first
plt.subplot(311)
plt.grid()
plt.plot(t, np.real(aa), label='a_tilda Real part')
plt.plot(t, np.imag(aa), label='a_tilda Imaginary part')
plt.plot(t, np.real(bb), label='b_tilda Real part')
plt.plot(t, np.imag(bb), label='b_tilda Imaginary part')
plt.xlabel('t')
plt.legend()
#second
plt.subplot(312)
plt.grid()
plt.plot(t, np.real(a), label='a real part')
plt.plot(t, np.imag(a), label='a imaginary part')
plt.plot(t, np.real(b), label='b real part')
plt.plot(t, np.imag(b), label='b imaginary part')
plt.xlabel('t')
plt.legend()
#third probability
plt.subplot(313)
plt.grid()
plt.plot(t, a*np.conj(a)+b*np.conj(b))

# show the plot
plt.show()