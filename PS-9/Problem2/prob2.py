import numpy as np
import matplotlib.pyplot as plt

#Constant coefficients:
R = 0.08 #[m]
V_0 = 100 #[m/s]
rho = 1.22 #[kg/m^3]
C = 0.47 #Unitless drag coefficient.
g = 9.81 #[m/s^2] Acceleration due to gravity.

#Collective drag term as a function of mass for part c).
def drag(m):
    return -np.pi*(R**2)*rho*C/(2*m) #A grouping of terms that arise later.

#Establishes an array of the 4 differential equations:
def cannonball(motion, m):
    x = motion[0]
    y = motion[1]
    Vx = motion[2]
    Vy = motion[3]

    x_prime = Vx
    y_prime = Vy
    Vx_prime = drag(m)*Vx*np.sqrt(Vx**2 + Vy**2)
    Vy_prime = -g + drag(m)*Vy*np.sqrt(Vx**2 + Vy**2)
    return np.array([x_prime,y_prime,Vx_prime,Vy_prime], float)



def trajectory(V_0, theta, m):
    x_array = []
    y_array = []
    Vx_0 = V_0*np.cos(theta)
    Vy_0 = V_0*np.sin(theta)
    motion = np.array([0,0,Vx_0,Vy_0], float) #Initializing x,y,Vx,Vy state array.

    for x_p in x_array:
        x_array.append(motion[0])
        y_array.append(motion[1])
        #Runge-Kutta 4th Order implimentation:
        #Application of RK4 in these Problems don't need normal time manipulation because
        #none of the equations are explicitly time-dependent.
        k1 = delta_t*cannonball(state, m)
        k2 = delta_t*cannonball(state + 0.5*k1, m)
        k3 = delta_t*cannonball(state + 0.5*k2, m)
        k4 = delta_t*cannonball(state + k3, m)
        state += (k1 + 2*k2 + 2*k3 + k4)/6
    return x_array, y_array




