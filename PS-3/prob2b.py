from random import random
import numpy as np
from matplotlib import pyplot as plt

# Initialization of radioisotope counts
Bi213 = 10000
Bi209 = 0 
Tl209 = 0 
Pb209 = 0 

# Half-lives of the different parent atoms (seconds)
tau_Bi213 = 2760
tau_Tl209 = 132
tau_Pb209 = 198

deltat = 1  # Time step
t_max = 20000  # Max decay time tracked
tpoints = np.arange(0, t_max, deltat)  # Time points

# Function to simulate decay of a parent isotope
def decay_process(tau, parent): 
    p = 1 - 2**(-deltat/tau)  # Decay probability within a time interval delta-t
    decay = 0  # Resets number of decays
    for i in range(parent):  # Random probability of decay for each existing parent atom
        if random() < p:
            decay += 1 
    parentloss = -decay  # Total decays over all parent atoms in one time interval
    daughtergain = decay
    return parentloss, daughtergain

# Simulate Bi213 decaying into either Pb209 or Tl209
def Bi213_decay():
    if random() < 0.9791:  # 97.91% probability of decaying into Pb209
        Bi_Pb = True
        Bi_Tl = False
    else:  # 2.09% probability of decaying into Tl209
        Bi_Pb = False
        Bi_Tl = True
    return Bi_Pb, Bi_Tl

# Initialize the count array for isotopes over time
isocount = np.zeros((int(t_max / deltat), 4), dtype=int)

# Function to simulate the decay process for all isotopes
def isodecays(): 
    # Initial population of Bi213
    isocount[0, 0] = Bi213
    isochange = np.zeros((int(t_max / deltat), 4), dtype=int)  # Change in isotope counts

    for t in range(1, len(tpoints)):
        # Decay process for Bi213
        bi213_decay_result = Bi213_decay()  # Store the result of the decay
        Bi_Pb = bi213_decay_result[0]
        Bi_Tl = bi213_decay_result[1]
        
        # Decay changes for each isotope
        pb_bi = decay_process(tau_Pb209, isocount[t-1, 2])  # Pb209 -> Bi209
        tl_pb = decay_process(tau_Tl209, isocount[t-1, 1])  # Tl209 -> Pb209
        bi_pb = decay_process(tau_Bi213, isocount[t-1, 0])  # Bi213 -> Pb209
        bi_tl = decay_process(tau_Bi213, isocount[t-1, 0])  # Bi213 -> Tl209
        
        # Update isotope changes
        isochange[t, :] = (
            bi_pb[0] + bi_tl[0],  # Change in Bi213 atoms
            bi_tl[1] + tl_pb[0],  # Change in Tl209 atoms
            bi_pb[1] + tl_pb[1] + pb_bi[0],  # Change in Pb209 atoms
            pb_bi[1]  # Change in Bi209 atoms
        )
        
        # Update isotope counts by adding changes from previous time step
        isocount[t, :] = isocount[t-1, :] + isochange[t, :]

    return isocount

# Example run to simulate the decay process
result = isodecays()

# Plot the decay over time
plt.plot(tpoints, result[:, 0], label="Bi213")
plt.plot(tpoints, result[:, 1], label="Tl209")
plt.plot(tpoints, result[:, 2], label="Pb209")
plt.plot(tpoints, result[:, 3], label="Bi209")
plt.xlabel("Time (s)")
plt.ylabel("Atom count")
plt.legend()
plt.title("Decay of Bi213 Isotope and Its Daughters")
plt.show()
plt.savefig('lkadjf')
