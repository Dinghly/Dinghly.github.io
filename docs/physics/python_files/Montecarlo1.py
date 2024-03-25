import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return np.exp(-x**2 - y**2)

def monte_carlo_integration_x_fixed(x, y_min, y_max, num_samples):
    y_samples = np.random.uniform(y_min, y_max, num_samples)
    integral_sum = np.sum(f(x, y_samples))
    integral_approximation = (y_max - y_min) * (integral_sum / num_samples)
    return integral_approximation

# Define the range for x
x_values = np.linspace(-5, 5, 100000)

# Compute f(x) for each x value using Monte Carlo integration
f_x_values = []
num_samples = 100000  # Number of Monte Carlo samples
y_min = -5  # Lower limit for y
y_max = 5   # Upper limit for y

for x in x_values:
    integral_approximation = monte_carlo_integration_x_fixed(x, y_min, y_max, num_samples)
    f_x_values.append(integral_approximation)

# Plot f(x)
plt.plot(x_values, f_x_values)
plt.title('Plot of f(x)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.show()
