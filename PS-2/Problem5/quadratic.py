import numpy as np

def quadratic(a,b,c):
    if b > 0:
        root1 = np.float64((-b - np.sqrt(b**2 - 4*a*c))/(2*a))
        root2 = np.float64(2*c/(-b - np.sqrt(b**2 - 4*a*c)))
    if b < 0:
        root1 = np.float64((-b + np.sqrt(b**2 - 4*a*c))/(2*a))
        root2 = np.float64(2*c/(-b + np.sqrt(b**2 - 4*a*c)))
    return root1,root2

