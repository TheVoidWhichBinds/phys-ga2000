import numpy as np

number = np.float64(100.98763) #saving original number with higher precision
number32 = np.float32(100.98763) #casting original number to float32 precision
def floattobits(number32):
    bytes_rep = np.frombuffer(number32.tobytes(), dtype=np.uint8) #converts number to bytes
    bits = np.unpackbits(bytes_rep) #converts bytes to bits
    return bits

bits = floattobits(number32)

def bitstofloat(bits):
    bytes_rep = np.packbits(bits).tobytes() #converts bits to bytes
    saved = np.frombuffer(bytes_rep, dtype = np.float32) #converts bytes to its float32 number
    return saved

saved = np.float64(bitstofloat(bits)[0]) #np.frombuffer produces a 1-element array ->
#its value must be taken in order to subtract it from the original number
difference = number - saved 
print("The stored 32-bit floating point representation is", saved)
print("The difference between float32 saved number and the original is", difference)

