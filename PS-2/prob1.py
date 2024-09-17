import numpy as np

number = np.float64(100.98763)
number32 = np.float32(100.98763)
def floattobits(number32):
    bytes_rep = np.frombuffer(number32.tobytes(), dtype=np.uint8)
    bits = np.unpackbits(bytes_rep)
    return bits

bits = floattobits(number32)

def bitstofloat(bits):
    bytes_rep = np.packbits(bits).tobytes()
    saved = np.frombuffer(bytes_rep, dtype = np.float32)
    return saved

saved = bitstofloat(bits)[0]

difference = number - saved
print(difference)

