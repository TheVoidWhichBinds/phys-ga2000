import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

#Brent method done via Python:
def f(x):
    return np.exp(x)*(x - 0.3)**2

brent_python = brentq(f,0,1)
print(brent_python)


#Brent method done 'manually':
def brent_method(f, a, b, c, tol=1e-5):
    #asigns function call to variables.
    fa = f(a)
    fb = f(b)
    fc = f(c)
    
    while abs(c - a) > tol:
        #parabola parameters.
        R = fb/fc
        S = fb/fa
        T = fa/fc
        P = S * [T*(R - T)*(c - b) - (1 - R)*(b - a)]
        Q = (T - 1)*(R - 1)*(S - 1)
        
        #division by zero/overflow prevention.
        if abs(Q) < 1e-10:
            Q = 1e-10
        
        # Calculate the new candidate point using the parabolic interpolation formula
        x_new = b + P / Q
        
        # Check if the new point is within the bracketing interval
        if x_new < a or x_new > c:
            # If not, fallback to bisection method
            x_new = (a + c) / 2
        
        # Evaluate the function at the new point
        f_new = f(x_new)
        
        # Update the bracketing points
        if f_new < fb:
            # If the new point improves the function value, update the bracket
            a, fa = b, fb
            b, fb = x_new, f_new
        else:
            # Otherwise, adjust the bracket to include the new point
            c, fc = x_new, f_new
        
    return b, fb

# Test the implementation of Brent's method on the given function
a = -2  # Lower bound of the interval
b = 0   # Initial midpoint
c = 2   # Upper bound of the interval

minimum_x, minimum_f = brent_method(f, a, b, c)

print(f"The minimum is at x = {minimum_x}, with a function value of {minimum_f}")
