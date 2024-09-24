import numpy as np
import matplotlib.pyplot as plt

N = 1000 #number of atoms.
tau = 3.053 #half-life constant in minutes.

uni = np.random.uniform(0,1,N) #uniform 1-D array of random numbers between 0 and 1.
t_decay = np.sort(-(tau/np.log(2))*np.log(1-uni)) #sorted decay times of N radioisotopes.

t = np.linspace(0,max(t_decay),N, dtype = int) #time axis.
Tl208 = np.zeros((N,1)) #initializing Tl208 population array.
#For-loop below: at each t value "time", populates the Tl208 array at 
#the i-th time step with the # of Tl208 which have not yet decayed:
for i,time in enumerate(t): 
    Tl208[i,0] = np.sum(t_decay[:] > time)

plt.plot(t,Tl208)
plt.title('Thallium-208 Decay')
plt.ylabel('Tl-208 Population')
plt.xlabel('Time (minutes)')
plt.savefig('Random_Decay')