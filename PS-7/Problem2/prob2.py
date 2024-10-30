import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import time


#function is defined:
def f(x):
    return np.exp(x) * (x - 0.3)**2

#scipy function that finds the minimum using Brent's method.
start_time = time.time()
result = minimize_scalar(f, method='brent')
end_time = time.time()
print(f"Using Brent's method via SciPy, the root is at x = {result.x}")
print(f"Time taken by SciPy's Brent method: {end_time - start_time:.6f} seconds\n")

#'manual' implementation of Brent's method:
def brent_method(f, a, b, c, tol=1e-5):
    #evaluating the function at the bounds and midpoint:
    fa = f(a)
    fb = f(b)
    fc = f(c)
    
    while abs(c - a) > tol: #iteratiions search for the minimum within smaller and smaller delta x until tolerance met.
        #parabola parameters.
        R = fb / fc
        S = fb / fa
        T = fa / fc
        P = S * (T * (R - T) * (c - b) - (1 - R) * (b - a))
        Q = (T - 1) * (R - 1) * (S - 1)
        
        #division by zero/overflow prevention.
        if abs(Q) < 1e-10:
            Q = 1e-10
        
        #calculates the new point using the parabolic interpolation formula.
        x_new = b + P / Q
        
        #checks if the new point is within the bracketing interval.
        if x_new < a or x_new > c:
            #if not, returns to bisection method.
            x_new = (a + c) / 2
        
        #evaluate the function at the new point.
        f_new = f(x_new)
        
        #updates the bracketing points:
        if f_new < fb:
            #if the new point improves the function value, the bracket is updated.
            a, fa = b, fb
            b, fb = x_new, f_new
        else:
            #otherwise, bracket updated to include the new point.
            c, fc = x_new, f_new
        
    return b, fb

#testing the implementation of Brent's method on the given function:
a = -2  #lower bound.
b = 0   #initial midpoint.
c = 2   #upper bound.

start_time = time.time()
minimum_x, minimum_f = brent_method(f, a, b, c)
end_time = time.time()
print(f"Using Brent's method via my own function, the root is at x = {minimum_x}")
print(f"Time taken by manual Brent method: {end_time - start_time:.6f} seconds")