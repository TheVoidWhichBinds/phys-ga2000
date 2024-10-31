import jax.numpy as jnp
from jax import grad, hessian
from jax.scipy.optimize import minimize

# Define the logistic function
def logistic(x, beta0, beta1):
    return 1 / (1 + jnp.exp(-(beta0 + beta1 * x)))

# Define the negative log-likelihood function
def negative_log_likelihood(params, x, y):
    beta0, beta1 = params
    p = logistic(x, beta0, beta1)
    # Add a small value to prevent log(0)
    eps = 1e-10
    return -jnp.sum(y * jnp.log(p + eps) + (1 - y) * jnp.log(1 - p + eps))

# Load or define your data for `x` (ages) and `y` (responses 0 or 1)
x_data = jnp.array([1,2,3,4,5])  # Ages
y_data = jnp.array([1,0,1,1,0])  # Responses (1 for "yes", 0 for "no")

# Initial guess for beta0 and beta1
initial_params = jnp.array([0.0, 0.0])

# Optimize to find the best beta0 and beta1
result = minimize(negative_log_likelihood, initial_params, args=(x_data, y_data))
beta0_opt, beta1_opt = result.x

# Compute the covariance matrix using the Hessian
hess = hessian(negative_log_likelihood)(jnp.array([beta0_opt, beta1_opt]), x_data, y_data)
cov_matrix = jnp.linalg.inv(hess)

# Formal errors are the square roots of the diagonal elements of the covariance matrix
errors = jnp.sqrt(jnp.diag(cov_matrix))

# Plot the results
import matplotlib.pyplot as plt

ages = jnp.linspace(min(x_data), max(x_data), 100)
probabilities = logistic(ages, beta0_opt, beta1_opt)

plt.scatter(x_data, y_data, label="Survey Data", alpha=0.5)
plt.plot(ages, probabilities, label="Logistic Model", color="red")
plt.xlabel("Age")
plt.ylabel("Probability of 'Yes'")
plt.legend()
plt.show()
