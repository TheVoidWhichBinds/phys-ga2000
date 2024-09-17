import numpy as np
import matplotlib as plt

#1: from lecture def get_bits(number) & def get_fbits then use formula and take difference

#2: 1+2^-23 is the smallest addition to 1
#np.finfo can get these values
#Largest number is 3.40282*10^38

#3: np.meshgrid
#import timeit  (timeit.timeit)

#4: 


#set dtype of ALL integers (32 bit or 64 bit) np.float32/np.float64
#you can create a python code 



import numpy as np

n = 10

d = np.arange(-n,n+1)
i,j,k = np.meshgrid(d,d,d)

origin = (i == 0) & (j == 0) & (k == 0)
M_ijk = ((-1)**np.abs(i+j+k)[~origin])/np.sqrt(i**2 + j**2 + k**2)[~origin]
M = M_ijk.sum()

print(M)