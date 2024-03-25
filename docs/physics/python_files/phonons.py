import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite
from mpl_toolkits.mplot3d import Axes3D

#global constant
N = 2
a = 1
kappa = 1
m = 1
h = 1
#Discrete fourier transformation or second quantization
#here I am working with j instead of k because k is not an integer but j is.
#assuming that k = 2*pi*j/N*a
size = 63896868
#Our space of interest
q1 = np.linspace(-size ,size ,10000)
#Second quantization
def Q1(q1, q2, j1):
    return q1*np.exp(-1j*(2*np.pi*j1/(N*a))) + q2*np.exp(-1j*(2*np.pi*j1/(N*a)))
def Q2(q1, q2, j2):
    return q1*np.exp(-1j*(2*np.pi*j2/(N*a))) + q2*np.exp(-1j*(2*np.pi*j2/(N*a)))
#dispersion relation with k
def w_k(j):
    #print(2*np.sqrt(kappa/m)*np.absolute(np.sin((2*np.pi*j)/(N*a))))
    return 2*np.sqrt(kappa/m)*np.absolute(np.sin((2*np.pi*j)/(2*N*a)))

def psi_j(Q, w, n):
    Q0 = np.sqrt(h/(m*w))
    A = 1/(np.sqrt((np.sqrt(np.pi))*(2**n)*(math.factorial(n))*Q0))
    _hermite = hermite(n)(Q/Q0)
    #print(Q0)
    #psi = w**(1/4)*np.exp(-w*Q**2)
    #hermite_Q = np.array([_hermite(_Q) for _Q in Q/Q0])
    gaussian = np.exp(-(Q**2)/(2*(Q0**2)))
    return A*_hermite*gaussian

# Prob = psi_j(Q1, w_k(1), 0)*psi_j(Q2, w_k(2), 0)

# Initialize array to store integral estimates for each q
integral_estimates = []
q2_min, q2_max = -size, size
# Number of random samples
num_samples = 10000
#perform monte carlo 
for q in q1:
    q2_samples = np.random.uniform(q2_min, q2_max, num_samples)

    f_values = (psi_j(np.real(Q1(q,q2_samples,1)), (w_k(1)),0)*psi_j(np.real(Q2(q,q2_samples, 2)), (w_k(2)),0))
    # Compute the integral estimate for this value of x
    integral_estimate = np.mean(f_values) * (q2_max - q2_min)
    integral_estimates.append(integral_estimate)

q2 = np.linspace(-size, size, 10000)
#plt.plot(q1, (psi_j(np.real(Q1(q1,q2,1)), (w_k(1)),3))**2)
plt.plot(q1, integral_estimates)
plt.xlabel('q1')
plt.ylabel('Integral Estimate')
plt.title('Monte Carlo Integration Result')
plt.grid(True)
plt.show()