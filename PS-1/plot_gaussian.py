import numpy as np
import matplotlib.pyplot as plt


o = 3 #standard deviation
u = 0 #mean
x = np.arange(-10,10,0.1) #gives inputs to the function

gaussian = (o*np.sqrt(2*np.pi))**(-1)*np.exp(-0.5*((x-u)/o)**2)

plt.plot(x,gaussian)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Normalized Gaussian Distribution")

plt.savefig('guassian.png')
