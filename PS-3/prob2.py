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
        return 1,0
    else:
        return 0,1
    

#initialization of 2D array which keeps track of isotope number (columns) vs time (rows)
isocount = np.zeros((int(t_max/deltat),4),dtype=int)
isocount[0,0] = Bi213
isocount[0,1] = Tl209
isocount[0,2] = Pb209 
isocount[0,3] = Bi209



#calls decay_process with global count of each isotope. I need the input 
def isodecays(): 
    isochange = np.zeros((int(t_max/deltat),4),dtype=int)
    for t in range(1,len(tpoints)):
        Pbpath,Tlpath = Bi213_decay()
        Bi_Tl = decay_process(tau_Bi213,isocount[t-1,0])
        Bi_Pb = decay_process(tau_Bi213,isocount[t-1,0])
        Tl_Pb = decay_process(tau_Tl209,isocount[t-1,1])
        Pb_Bi = decay_process(tau_Pb209,isocount[t-1,2])
        
        isochange[t,:] = (Pbpath*Bi_Pb[0] + Tlpath*Bi_Tl[0], Tlpath*Bi_Tl[1] + Tl_Pb[0], Pbpath*Bi_Pb[1] + Tl_Pb[1] + Pb_Bi[0], Pb_Bi[1])
        isocount[t,:] = isocount[t-1,:] + isochange[t,:]
    return isocount

isocount = isodecays()


fig, major = plt.subplots()
plt.title('Bismuth-213 Decay', fontsize=17)
plt.xlabel('Time (seconds)', fontsize=12)
plt.ylabel('Radioisotope Population', fontsize=12)
plt.plot(tpoints,isocount[:,0],label = 'Bi213', color = 'r')
plt.plot(tpoints,isocount[:,3],label = 'Bi209', color = 'b')

minor = major.twinx()
minor.set_ylim(0,1000)
plt.plot(tpoints,isocount[:,1],label = 'Tl209', color = 'g')
plt.plot(tpoints,isocount[:,2],label = 'Pb209', color = 'm')

#LEGEND NEEDED

plt.savefig('isocount')
