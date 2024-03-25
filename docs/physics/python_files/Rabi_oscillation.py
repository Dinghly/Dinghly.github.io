import numpy as np
import matplotlib.pyplot as plt
from math import *

# Constants and parameters
w = 5 # Frequency for x(t) and y(t)
Omega0 = 5 #eB0
Omega1 = 1
Delta = w-Omega0
Omega = (Delta**2 + Omega1**2)**.5
w1 = Omega/Omega1
w2 = Delta/Omega1
T = 2*pi/Omega
t = np.linspace(0, 1*T, 100000)
# state vector
aa = ((w1+w2)/(2*w1))*np.exp(1j*Omega*t/2)+((w1-w2)/(2*w1))*np.exp(-1j*Omega*t/2)
bb = -(w1-w2)*((w1+w2)/(2*w1))*np.exp(1j*Omega*t/2)+(w1+w2)*((w1-w2)/(2*w1))*np.exp(-1j*Omega*t/2)

a = aa*np.exp(-1j*w*t/2)
b = bb*np.exp(1j*w*t/2)
#Coordinatizing expectation value of Spin operator
x = (np.conj(a)*b)+(np.conj(b)*a)
y = 1j*((-np.conj(a)*b)+(np.conj(b)*a))
z = (np.conj(a)*a)-(np.conj(b)*b)
# Create 3D plots
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z)

ax.axis('equal')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)

# Set labels for the axes
ax.set_xlabel('X(t)')
ax.set_ylabel('Y(t)')
ax.set_zlabel('Z(t)')