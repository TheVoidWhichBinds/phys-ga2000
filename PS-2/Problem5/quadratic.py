import numpy as np

def quadratic(a,b,c): #takes values of quadratic formula and solves for roots.
    discriminant = np.sqrt(b**2 - 4*a*c)
    if 4*a*c > b**2: 
        print("Imaginary solution")
    elif b == 0:
        if a*c > 0:
            print("Imaginary solution")
        elif a*c == 0:
            print("Trivial solution")
        else:
            root1 = np.float64(2*np.sqrt(a*c)/(2*a))
            root2 = np.float64(-2*np.sqrt(a*c)/(2*a))
    if b > 0:
        root1 = np.float64((-b - discriminant)/(2*a))
        root2 = np.float64(2*c/(-b - discriminant))
    elif b < 0:
        root1 = np.float64((-b + discriminant)/(2*a))
        root2 = np.float64(2*c/(-b + discriminant))
    
    return root1,root2


