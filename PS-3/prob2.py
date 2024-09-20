from random import random
import numpy as np
from matplotlib import pyplot as plt

#initialization of radioisotope counts:
Bi213 = 10000
Bi209 = 0 
Tl209 = 0 
Pb209 = 0 
#half-lives of the different parent atoms(seconds):
tau_Bi213 = 2760
tau_Tl209 = 132
tau_Pb209 = 198

deltat = 1 #time step.
t_max = 20000 #max decay time tracked.
tpoints = np.arange(0,t_max,deltat)

#I need a function to take in an 2D array with columns of isotope and rows of time blocks ->
#with elements that are the count of each atom at a moment in time.

def decay_process(tau,parent): 
    p = 1 - 2**(-deltat/tau) #decay probability within a time interval delta-t.
    decay = 0 #resets number of decays.
    for i in range(parent): #random probability of decay for each existing parent atom.
         if random() < p:
            decay += 1 
    parentloss = -decay #adds total decays over all parent atoms in one time interval.
    daughtergain = decay
    return parentloss, daughtergain


def Bi213_decay():
    if random()<0.9791:
        Bi_Pb = True
        Bi_Tl = False
    else:
        Bi_Pb = False
        Bi_Tl = True
    return Bi_Pb, Bi_Tl

isocount = np.zeros((int(t_max/deltat),4),dtype=int)

Pb_Bi = decay_process(tau_Pb209,isocount[t-1,2],[t-1,3])
Tl_Pb = decay_process(tau_Tl209,isocount[t-1,1],[t-1,2])
Bi_Pb = decay_process(tau_Bi213,isocount[t-1,0],[t-1,2])
Bi_Tl = decay_process(tau_Bi213,isocount[t-1,0],[t-1,1])

#calls decay_process with global count of each isotope. I need the input 
def isodecays(): 
    isocount[0,0] = Bi213
    isochange = np.zeros((int(t_max/deltat),4),dtype=int)

    for t in tpoints:
        isochange[t,:] = (Bi213_decay()[0]+Bi213_decay()[1], #change in Bi213 atoms.
                        Bi213_decay()[1]+Tl_Pb[0], #change in Ti209 atoms.
                        Bi213_decay()[1]+Tl_Pb[1]+Pb_Bi[0] #change in Pb209 atoms.
                        Pb_Bi[1]) #change in Bi209 atoms.
        isocount[t,:] = isocount[t-1,:] + isochange[t,:]
    
    return isocount


        



    


