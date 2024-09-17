import numpy as np

#initialization of quadratic formulae inputs.
a = 0.001
b = 1000
c = 0.001

#All roots given the two versions of the formula.
xpos = np.float64((-b + np.sqrt(b**2 - 4*a*c))/(2*a))
xneg = np.float64((-b - np.sqrt(b**2 - 4*a*c))/(2*a))
invpos = np.float64(2*c/(-b + np.sqrt(b**2 - 4*a*c)))
invneg = np.float64(2*c/(-b - np.sqrt(b**2 - 4*a*c)))

print(xpos)
print(xneg)
print(invpos)
print(invneg)

#xneg and invneg yield more accurrate solutions. 

