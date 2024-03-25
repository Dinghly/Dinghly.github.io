# import numpy as np
# import matplotlib.pyplot as plt

# def gaussian_3d(x, y, z, sigma=1):
#     """3-dimensional Gaussian function."""
#     return np.exp(-(x**2 + y**2 + z**2) / (2 * sigma**2))

# def accept_reject_integration(x_range, sigma=1, samples_per_x=10000):
#     """Integrate the 3D Gaussian over y and z using the von Neumann accept-reject technique."""
#     yz_bounds = [-4*sigma, 4*sigma]
#     integrated_values = []
    
#     for x in x_range:
#         accepted_samples = 0
#         for _ in range(samples_per_x):
#             # Generate y, z from the proposal uniform distribution
#             y_sample = np.random.uniform(*yz_bounds)
#             z_sample = np.random.uniform(*yz_bounds)
#             # Calculate the acceptance probability
#             f_val = gaussian_3d(x, y_sample, z_sample, sigma)
#             M = 1  # Maximum of f(x, y, z) for a normalized Gaussian
#             q_val = 1 / ((yz_bounds[1] - yz_bounds[0]) ** 2)  # Proposal distribution is uniform
#             acceptance_prob = f_val / (M * q_val)
            
#             # Generate a uniform random number to decide if we accept the sample
#             if np.random.rand() < acceptance_prob:
#                 accepted_samples += 1
        
#         # Estimate the integral as the fraction of accepted samples times the area of the proposal distribution
#         integral_estimate = (accepted_samples / samples_per_x) * ((yz_bounds[1] - yz_bounds[0]) ** 2)
#         integrated_values.append(integral_estimate)
        
#     return np.array(integrated_values)

# # Define the range of x values
# x_range = np.linspace(-4, 4, 100)

# # Perform Monte Carlo Integration using the accept-reject method
# integrated_values_ar = accept_reject_integration(x_range)

# # Plot the integrated function
# plt.plot(x_range, integrated_values_ar, label='Accept-Reject Integration')
# plt.xlabel('x')
# plt.ylabel('Integrated value over y and z')
# plt.title('Integration of 3D Gaussian over y and z\n(Accept-Reject Technique)')
# plt.legend()
# plt.grid(True)
# plt.show()

##############################################################################################3
# import numpy as np
# import matplotlib.pyplot as plt

# # Define the 3-dimensional Gaussian function
# def gaussian_3d(x, y, z, sigma=1):
#     return np.exp(-(x**2 + y**2 + z**2) / (2 * sigma**2))

# # Direct Monte Carlo integration
# def direct_monte_carlo_integration(x_range, sigma=1, samples=10000):
#     yz_bounds = [-10*sigma, 10*sigma]
#     integrated_values = []
    
#     for x in x_range:
#         y_samples = np.random.uniform(yz_bounds[0], yz_bounds[1], samples)
#         z_samples = np.random.uniform(yz_bounds[0], yz_bounds[1], samples)
#         function_samples = gaussian_3d(x, y_samples, z_samples, sigma)
#         integral_estimate = np.mean(function_samples) * (yz_bounds[1] - yz_bounds[0])**2
#         integrated_values.append(integral_estimate)
        
#     return np.array(integrated_values)

# # Accept-reject Monte Carlo integration
# def accept_reject_mc_integration(x_range, sigma=1, total_samples=10000):
#     yz_bounds = [-10*sigma, 10*sigma]
#     M = 1  # Maximum value of the Gaussian function for normalized conditions
#     integrated_values = []
    
#     for x in x_range:
#         accepted = 0
#         for _ in range(total_samples):
#             y = np.random.uniform(*yz_bounds)
#             z = np.random.uniform(*yz_bounds)
#             if np.random.uniform(0, M) < gaussian_3d(x, y, z, sigma):
#                 accepted += 1
#         integral_estimate = (accepted / total_samples) * (yz_bounds[1] - yz_bounds[0])**2
#         integrated_values.append(integral_estimate)
    
#     return np.array(integrated_values)

# # Range of x values
# x_range = np.linspace(-10, 10, 10000)

# # Perform both Monte Carlo Integration methods
# integrated_values_direct = direct_monte_carlo_integration(x_range)
# integrated_values_accept_reject = accept_reject_mc_integration(x_range)

# # Plotting the results
# plt.figure(figsize=(10, 6))
# plt.plot(x_range, integrated_values_direct, label='Direct Monte Carlo Integration', color='blue')
# plt.plot(x_range, integrated_values_accept_reject, label='Accept-Reject Monte Carlo Integration', color='red')
# plt.xlabel('x')
# plt.ylabel('Integrated value over y and z')
# plt.title('Comparison of Monte Carlo Integration Methods')
# plt.legend()
# plt.grid(True)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

def gaussian_3d(x, y, z, sigma=1):
    """3-dimensional Gaussian function."""
    return np.exp(-(x**2 + y**2 + z**2) / (2 * sigma**2))

def accept_reject_mc_integration(x_range, sigma=1, total_samples=10000):
    """Accept-reject Monte Carlo integration of the 3D Gaussian over y and z for a range of x."""
    yz_bounds = [-3*sigma, 3*sigma]
    M = 1  # Maximum value of f(x, y, z) for sigma=1 and x, y, z = 0
    integrated_values = []
    
    for x in x_range:
        accepted_samples = 0
        for _ in range(total_samples):
            y = np.random.uniform(*yz_bounds)
            z = np.random.uniform(*yz_bounds)
            # Proposal density is uniform, so q(y, z) is constant and cancels out in the ratio
            f_val = gaussian_3d(x, y, z, sigma)
            if np.random.uniform(0, M) < f_val:
                accepted_samples += 1
        # Estimate the integral as the ratio of accepted points to total points, times the area of the proposal distribution
        integral_estimate = (accepted_samples / total_samples) * (yz_bounds[1] - yz_bounds[0])**2
        integrated_values.append(integral_estimate)
    
    return np.array(integrated_values)

# Define the range of x values
x_range = np.linspace(-3, 3, 100)

# Perform accept-reject Monte Carlo Integration
integrated_values_accept_reject = accept_reject_mc_integration(x_range)

# Normalize the integrated values to have a maximum amplitude of 1
normalized_integrated_values_ar = integrated_values_accept_reject / np.max(integrated_values_accept_reject)

# Plotting the results
plt.figure(figsize=(8, 6))
plt.plot(x_range, normalized_integrated_values_ar, label='Normalized Accept-Reject Monte Carlo Integration', color='green')
plt.xlabel('x')
plt.ylabel('Normalized Integrated value over y and z')
plt.title('Normalized Accept-Reject Monte Carlo Integration of 3D Gaussian over y and z')
plt.legend()
plt.grid(True)
plt.show()
