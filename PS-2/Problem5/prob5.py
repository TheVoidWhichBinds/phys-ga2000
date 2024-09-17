import numpy as np

a = 0.001
b = 1000
c = 0.001

xpos = np.float64((-b + np.sqrt(b**2 - 4*a*c))/(2*a))
xneg = np.float64((-b - np.sqrt(b**2 - 4*a*c))/(2*a))
invpos = np.float64(2*c/(-b + np.sqrt(b**2 - 4*a*c)))
invneg = np.float64(2*c/(-b - np.sqrt(b**2 - 4*a*c)))

print(xpos)
print(xneg)
print(invpos)
print(invneg)

#xneg and invneg yield more accurrate solutions. 

