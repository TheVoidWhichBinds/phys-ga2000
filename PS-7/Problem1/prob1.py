import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq


def Lagrange(r_0,R,M,m):
    acc = 1e-6
    r_delta = 0
    r = r_0
    while abs(r_delta) < acc:
        f = (M/r**2) - (r*M/R**3) - (m/(R-r)**2)
        f_prime = (-2*M/r**3) - (M/R**3) - (2*m/(R-r)**3)
        r_delta = f/f_prime
        r -= r_delta
    return r



