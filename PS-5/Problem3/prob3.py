import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('signal.dat', sep='\s+', header=0) 

plt.figure()
plt.figure(figsize=(10, 6)) 
plt.title('Signal', fontsize = 20)
plt.ylabel('Amplitude',fontsize = 14)
plt.xlabel('Time',fontsize = 14)
plt.scatter(data['time'], data['signal'])
plt.savefig('Signal')








t = data['time'].values
a = data['signal'].values

# Center the time values (scaling: subtract mean and divide by standard deviation)
t_mean = np.mean(t)
t_std = np.std(t)
t_scaled = (t - t_mean) / t_std  # This improves numerical stability for SVD

# Construct the design matrix for a third-order polynomial
# We are fitting y = c0 + c1*x + c2*x^2 + c3*x^3
A = np.column_stack([np.ones_like(t_scaled), t_scaled, t_scaled**2, t_scaled**3])

# Perform the Singular Value Decomposition (SVD)
U, s, Vt = np.linalg.svd(A, full_matrices=False)
S_inv = np.diag(1/s)

# Compute the pseudoinverse of the design matrix
A_pseudo_inv = Vt.T @ S_inv @ U.T

# Solve for the coefficients (c0, c1, c2, c3)
coefficients = A_pseudo_inv @ a

# Create a function for the best-fit polynomial model
def model(x, coeffs):
    return coeffs[0] + coeffs[1] * x + coeffs[2] * x**2 + coeffs[3] * x**3

# Generate model predictions using the scaled time data
a_fit = model(t_scaled, coefficients)

# (c) Calculate residuals and compare with measurement uncertainties
residuals = a - a_fit

# Plot the fit and residuals
plt.figure(figsize=(8, 6))

# Plot the original data and the fitted polynomial
plt.subplot(2, 1, 1)
plt.plot(t, a, 'o', label='Signal Data')
plt.plot(t, a_fit, label='Best Fit (3rd order polynomial)', color='red')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.title('Best 3rd Order Polynomial Fit to Signal')
plt.legend()

# Plot the residuals
plt.subplot(2, 1, 2)
plt.plot(t, residuals, 'o', label='Residuals')
plt.axhline(2.0, color='red', linestyle='--', label='Measurement uncertainty')
plt.axhline(-2.0, color='red', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Residuals')
plt.title('Residuals of Fit')
plt.legend()

plt.tight_layout()
plt.savefig('SVD')

# Check if the residuals exceed the uncertainty
max_residual = np.max(np.abs(residuals))
print(f"Maximum Residual: {max_residual:.2f}")

if max_residual > 2.0:
    print("The residuals exceed the measurement uncertainty, suggesting the model is not a good fit.")
else:
    print("The model fits within the measurement uncertainty.")
