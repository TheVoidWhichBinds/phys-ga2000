import numpy as np
import matplotlib.pyplot as plt


o = 3 #standard deviation
u = 0 #mean
x = np.arange(-10,10,0.1) #gives inputs to the function

gaussian = (o*np.sqrt(2*np.pi))**(-1)*np.exp(-0.5*((x-u)/o)**2)

plt.plot(x,gaussian)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Normalized Gaussian Distribution")
plt.text(-8, 0.11, '$\\sigma = 3$')
plt.text(-8, 0.10, '$\\mu = 0$')

plt.savefig('guassian.png')
