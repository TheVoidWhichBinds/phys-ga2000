import numpy as np


def addrez32(): #finds the min value that can be added to 1 with different float casting.
    prerez32 = None #saves an iteration before the next loop.
    for p in range(1,20): #iterates along some arbitrary range.
        flt32 = np.float32(1 + 10**(-p)) #1 added to iteratively smaller numbers.
        if flt32 == 1.0: #once the addition is indestinguishable from 1.0, ->
            return prerez32 #the previous iteration value is returned.
        prerez32 = flt32 #if the resolution is still intact, the new pre-loop value is saved.
     
def addrez64(): #same thing as addrez32 but withe float64
    prerez64 = None
    for p in range(1,20):
        flt64 = np.float64(1 + 10**(-p))
        if flt64 == 1.0:
            return prerez64
        prerez64 = flt64

#puts return values into scientific notation
flt32rez = np.format_float_scientific(1-addrez32(), precision=3) 
flt64rez = np.format_float_scientific(1-addrez64(), precision=3)

#uses np.finfo and .tiny to acertain the minimum value of any value cast to float32 and float64.
min32 = np.finfo(np.float32).tiny 
min64 = np.finfo(np.float64).tiny
max32 = np.finfo(np.float32).max
max64 = np.finfo(np.float64).max

print("The smallest number that can be added to 1 using np.float32 is", flt32rez)
print("The smallest number that can be added to 1 using np.float64 is", flt64rez)
print("Minimum value before underflow (float32) is ",min32)
print("Minimum value before underflow (float64) is ",min64)
print("Maximum value before overflow (float32) is", max32)
print("Maximum value before overflow (float64) is", max64)