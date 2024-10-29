import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq


def Lagrange(r_0,R,m,M): #function inputs: initial guess, distance beteween centers of planetary bodies, smaller planetary mass, larger planetary mass.
    acc = 1e-4 #target accuracy.
    r_delta = 1 #initialization of the delta.
    r_p = r_0/R #rescaling radius.
    m_p = m/M #rescaling mass.
    while abs(r_delta) > acc: #keeps the iteration finding the root/Lagrange point until the specified minimum accuracy.
        f = (1/r_p**2) - (r_p) - (m_p/(1-r_p)**2) #Fnet = ma with rescaling.
        f_p = (-2/r_p**3) - (1) - (2*m_p/(1-r_p)**3) #Derivative of the above function.
        r_delta = f/f_p
        r_p -= r_delta #Newton-Raphson iteration formula.
        return f"{r_p * R:.3e}"

print('The Lagrange point between the Earth and Moon is ', Lagrange(3e8,3.844e8,7.438e22,5.972e24), 'meters from the center of the Earth')
print('The Lagrange point between the Sun and Earth is ', Lagrange(1.3e11,1.496e11,5.972e24,1.989e30), 'meters from the center of the Sun')
print('The Lagrange point of a Jupiter-mass planet orbiting the Sun at the distance of Earth is ', Lagrange(1.3e8,1.496e11,1.898e27,1.989e30), 'meters from the center of the Sun')