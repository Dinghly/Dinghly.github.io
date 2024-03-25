import numpy as np
import matplotlib.pyplot as plt

# Define the function
def f(x, y, z, w):
    return np.exp(-x**2) * np.exp(-y**2) * np.exp(-z**2) * np.exp(-w**2)

# Define the integration limits
x_min, x_max = -5, 5
y_min, y_max = -5, 5
z_min, z_max = -5, 5
w_min, w_max = -5, 5

# Number of random samples
num_samples = 100000

# Generate random samples for x
x_values = np.linspace(x_min, x_max, 100)  # Values of x for plotting
#x_samples = np.random.uniform(x_min, x_max, num_samples)

# Initialize array to store integral estimates for each x
integral_estimates = []

# Perform Monte Carlo integration for each value of x
for x in x_values:
    # Generate random samples for y, z, and w
    y_samples = np.random.uniform(y_min, y_max, num_samples)
    z_samples = np.random.uniform(z_min, z_max, num_samples)
    w_samples = np.random.uniform(w_min, w_max, num_samples)
    
    # Evaluate the function for each sample
    f_values = f(x, y_samples, z_samples, w_samples)
    
    # Compute the integral estimate for this value of x
    integral_estimate = np.mean(f_values) * (y_max - y_min) * (z_max - z_min) * (w_max - w_min)
    integral_estimates.append(integral_estimate)

# Plot the result of integration
plt.plot(x_values, integral_estimates)
plt.xlabel('x')
plt.ylabel('Integral Estimate')
plt.title('Monte Carlo Integration Result')
plt.grid(True)
plt.show()
