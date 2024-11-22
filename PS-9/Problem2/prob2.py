import numpy as np
import matplotlib.pyplot as plt

#Constant coefficients:
R = 0.08 #[m]
V_0 = 100 #[m/s]
rho = 1.22 #[kg/m^3]
C = 0.47 #Unitless drag coefficient.
g = 9.81 #[m/s^2] Acceleration due to gravity.
theta = np.pi/6 #[radians -> 30 degrees] (Counter-clockwise with respect to the horizontal).

#Arbitrary time parameters:
t_0 = 0
t_f = 7 #[s]
N = 100 #Number of time steps.
delta_t = (t_f - t_0)/N
t = np.arange(t_0, t_f, delta_t)
#------------------------------------------------------------------------------------------------



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



def trajectory(m):
    x_array = []
    y_array = []
    Vx_0 = V_0*np.cos(theta)
    Vy_0 = V_0*np.sin(theta)
    motion = np.array([0,0,Vx_0,Vy_0], float) #Initializing x,y,Vx,Vy state array.

    for t_p in t:
        x_array.append(motion[0])
        y_array.append(motion[1])
        #Runge-Kutta 4th Order implimentation:
        #Application of RK4 in these Problems don't need normal time manipulation because
        #none of the equations are explicitly time-dependent.
        k1 = delta_t*cannonball(motion, m)
        k2 = delta_t*cannonball(motion + 0.5*k1, m)
        k3 = delta_t*cannonball(motion + 0.5*k2, m)
        k4 = delta_t*cannonball(motion + k3, m)
        motion += (k1 + 2*k2 + 2*k3 + k4)/6
    return x_array, y_array
#-------------------------------------------------------------------------------------------------



#Plotting trajectory:
plt.figure()
plt.subplot()
plt.title('Mass Effect on Cannonball Trajectory')
plt.ylim(-1,60)
plt.xlabel('X Position [meters]')
plt.ylabel('Y Position [meters]')
colors = ['r', 'orange', 'yellow', 'g', 'b', 'purple', 'c', 'brown']
for c in range(0,8,1):
    plt.plot(trajectory(1*2**c)[0], trajectory(1)[1], color = colors[c], label = f'mass = {1*2**c}kg')
plt.legend(loc='upper right', fontsize=8) 
plt.savefig('Mass_Effect')


#Plotting the maximum x-displacement at a given angle and initial velocity as a function of mass.
plt.figure()
plt.title('Mass Effect on Cannonball Trajectory')
plt.xlabel('Cannonball Mass [kg]')
plt.ylabel('Maximum Horizontal Displacement [meters]')

zero = 1
m_var = np.arange(1, 10000, 100)
mass_values = []
x_max_values = []

for mass in m_var:
    y_array = trajectory(mass)[1]  # Extract the y-array (vertical positions)
    for i in range(10, len(y_array) - 10):  # Loop through the y_array to find where y = 0
        if np.abs(y_array[i]) < zero:  # When y reaches 0 (or close to 0)
            x_max_index = i  # Store the index where y = 0
            mass_values.append(mass)  # Add mass to the list
            x_max_values.append(trajectory(mass)[0][x_max_index])  # Add the corresponding x value
            print(mass_values, x_max_values)
        else:
            break

plt.scatter(mass_values, x_max_values, color='m', label='Max Horizontal Displacement')
plt.legend()

plt.savefig('Mass_Effect2')

