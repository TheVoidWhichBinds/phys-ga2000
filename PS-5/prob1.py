import numpy as np
import math
import matplotlib.pyplot as plt
from jax import autodiff

def central_difference(x):

    def f(x):
        y = 1 + 0.5*math.tanh(2*x)
        return y
    
    dx = np.float32(1e-6)
    dfdx = (f(x+dx/2) - f(x-dx/2))/dx
    return dfdx


plt.figure(figsize=(10, 6)) 
plt.title('Derivative Methods', fontsize = 20)
plt.ylabel(r'Derivative $\frac{df}{dx}$',fontsize = 14)
plt.xlabel('x',fontsize = 14)

x = np.linspace(-2,2,100)
plt.scatter(x,[central_difference(xi) for xi in x], label='Central Difference dx=1e-6', color = 'b', s = 7)
plt.plot(x,[1/(np.cosh(2*xi)**2) for xi in x], label = 'Analytic Derivative', color = 'r')
plt.savefig('Derivative_Methods')