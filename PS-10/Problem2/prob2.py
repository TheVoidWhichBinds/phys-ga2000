import numpy as np
import matplotlib.pyplot as plt

#Parameters
m = np.float32(9.109E-31) #[kg].
L = np.float32(1E-8) #Box length [meters].
x_0 = L/2 #[meters].
sigma = 1E-10 #[meters].
kappa = 5E10 #[m^-1].
N = 1000 ##Spatial slices.
x_step = L/N #Spatial step size [meters].
t_step = 1E-18 #Time step size [seconds].
x = np.arange(0,L,x_step)
