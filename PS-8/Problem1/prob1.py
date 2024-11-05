import jax.numpy as jnp
from jax import hessian
from jax.scipy.optimize import minimize
import matplotlib.pyplot as plt  
import pandas as pd

#Unpacking the data.
data = pd.read_csv('survey.csv') 
age = jnp.array(data.iloc[:, 0].values)    
answer = jnp.array(data.iloc[:, 1].values) 


#Probability function defined.
def logistic(age, beta0, beta1): #Takes x, and betas to be optimized.
    return 1 / (1 + jnp.exp(-(beta0 + beta1 * age)))

#Defines the negative log-likelihood function.
def negative_log_likelihood(params, age, answer):
    beta0, beta1 = params
    p = logistic(age, beta0, beta1) #Calling probability func.
    eps = 1e-10 #Small value added prevent log(0).
    return -jnp.sum(answer * jnp.log(p + eps) + (1 - answer) * jnp.log(1 - p + eps))



#Initial guess for beta0 and beta1.
initial_params = jnp.array([-0.2, 1.1])
#Optimization to find the best beta0 and beta1 based on initial guess.
optimized_params = minimize(negative_log_likelihood, initial_params, args=(age, answer), method='BFGS')
beta0_opt, beta1_opt = optimized_params.x

#Covariance matrix computed using the Hessian.
hess = hessian(negative_log_likelihood)(jnp.array([beta0_opt, beta1_opt]), age, answer)
cov_matrix = jnp.linalg.inv(hess)

#Formal errors are the square roots of the diagonal elements of the covariance matrix.
errors = jnp.sqrt(jnp.diag(cov_matrix))
print(f'The error ')


ages = jnp.linspace(min(age), max(age)) #graph x-axis
probability = logistic(ages, beta0_opt, beta1_opt) #probability function with optimized beta values.
plt.figure(figsize=(8, 6))
plt.scatter(age, answer, label='Observed Data', color='blue', alpha=0.6)
plt.plot(ages, probability, label='Logistic Model', color='r')
plt.xlabel('Age (years)', fontsize =12)
plt.ylabel("Probability of 'Yes'", fontsize = 12)
plt.legend()
plt.title("'Be Kind, Rewind' Phrase Recognition", fontsize = 14)
plt.savefig('Max_Likelihood.png')

