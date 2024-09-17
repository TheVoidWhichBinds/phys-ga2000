import numpy as np

def addrez32():
    prerez32 = None
    for p in range(1,20):
        flt32 = np.float32(1 + 10**(-p))
        if flt32 == 1.0:
            return prerez32
        prerez32 = flt32
     
def addrez64():
    prerez64 = None
    for p in range(1,20):
        flt64 = np.float64(1 + 10**(-p))
        if flt64 == 1.0:
            return prerez64
        prerez64 = flt64

flt32rez = np.format_float_scientific(1-addrez32(), precision=3)
flt64rez = np.format_float_scientific(1-addrez64(), precision=3)
#print("The smallest number that can be added to 1 using np.float32 is", flt32rez)
#print("The smallest number that can be added to 1 using np.float64 is", flt64rez)

def minimum32():
    premin32 = None
    for p in range(40):
        flt32 = np.float32(10**(-p))
        if np.abs(flt32) < np.finfo(np.float32).tiny:
            return premin32
        premin32 = flt32

def minimum64():
    premin64 = None
    for p in range(400):
        flt64 = np.float64(10**(-p))  # Use 64-bit float
        if np.abs(flt64) < np.finfo(np.float64).tiny:  # Check against 64-bit tiny value
            return premin64
        premin64 = flt64

min32 = np.format_float_scientific(minimum32(), precision=3)
min64 = np.format_float_scientific(minimum64(), precision=3)
#print("Minimum value before underflow (float32) is ",min32)
#print("Minimum value before underflow (float64) is ",min64)

def maximum32():
    premax32 = None
    for p in range(40):  
        flt32 = np.float32(10**p)
        if np.abs(flt32) > np.finfo(np.float32).max:  
            return premax32
        premax32 = flt32

def maximum64():
    premax64 = None
    for p in range(309):  
        flt64 = np.float64(10**p)  
        if np.abs(flt64) > np.finfo(np.float64).max: 
            return premax64
        premax64 = flt64

#max32 = np.format_float_scientific(maximum32(), precision=3)
max64 = np.format_float_scientific(maximum64(), precision=3)
#print("Maximum value before overflow (float32) is", max32)
print("Maximum value before overflow (float64) is", max64)






