import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Define the size of the lattice (N x N)
N = 100


# Initialize the lattice with random spins (+1 or -1)
lattice = np.random.choice([-1, 1], size=(N, N))

# Set the boundary values to 0
lattice[0, :] = 0  # Top row
lattice[:, 0] = 0  # Left column
lattice[-1, :] = 0  # Bottom row
lattice[:, -1] = 0  # Right column

# Define the interaction energy (J) and temperature (T)
J = 1.0
T = 5.0  # Temperature in units where Boltzmann constant k_B = 1

# Define a custom colormap for the spins
cmap = mcolors.ListedColormap(['blue', 'white', 'red'])
bounds = [-1.5, -0.5, 0.5, 1.5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

#plt.figure(figsize =(30,30))
plt.imshow(lattice, cmap=cmap, norm=norm)
plt.colorbar(ticks=[-1, 1], label='Spin')
plt.title('2D Ising Model initial state at T = {}'.format(T))
plt.show()

def calculate_energy(i, j):
    """Calculate the energy of interaction at site (i, j) with its neighbors."""
    energy = 0
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        if 0 <= i+di < N and 0 <= j+dj < N:
            energy -= J * lattice[i, j] * lattice[i+di, j+dj]
    return energy
print(calculate_energy(0,1))

# Metropolis algorithm
for step in range(1000000):  # Number of steps
    i, j = np.random.randint(1, N, size=2)  # Random site
    delta_E = -2 * calculate_energy(i, j)
    if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
        lattice[i, j] *= -1  # Flip the spin
Nup = np.sum(lattice == 1)
Ndown = np.sum(lattice == -1)
print(Nup)
print(Ndown)
      
    #if step%500==0:
# Plot the final lattice
#plt.figure(figsize =(30,30))
plt.imshow(lattice, cmap=cmap, norm=norm)
plt.colorbar(ticks=[-1, 1], label='Spin')
plt.title('2D Ising Model final state at T = {}'.format(T))
plt.show()