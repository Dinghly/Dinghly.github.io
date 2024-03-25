import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import *
from math import *

def omega(k):
    return 2*np.sqrt(kappa/m)*np.absolute(np.sin(k*a/2))

a = 1
kappa = 1
m = 1
kspace = np.arange(-2*2*np.pi/a,2*2*np.pi/a,0.01)

plt.grid()
plt.plot(kspace, omega(kspace), label = 'Dispersion relation of 1D chain')
plt.xlim(-2*2*np.pi/a,2*2*np.pi/a)
plt.ylim(0,2.3)
plt.xlabel('k')
plt.ylabel('w')
plt.legend(loc='upper right')

plt.show()