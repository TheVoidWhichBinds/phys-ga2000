import numpy as np
import matplotlib.pyplot as plt


#Defining temporal bounds & step size:
t_0 = 0
t_f = 50
N = 1000
delta_t = (t_f - t_0)/N
t = np.arange(t_0, t_f, delta_t)


#Runge-Kutta function that takes in initial position and velocity and outputs their time evolutions.
def xv_evolution(x_0, v_0, oscillator):
    x_array = []
    v_array = []
    state = np.array([x_0,v_0], float) #Initializing x,v state array.

    for t_p in t:
        x_array.append(state[0])
        v_array.append(state[1])
        
        #Runge-Kutta 4th Order implimentation:
        #Application of RK4 in these Problems don't need normal time manipulation because
        #none of the equations are explicitly time-dependent.
        k1 = delta_t*oscillator(state)
        k2 = delta_t*oscillator(state + 0.5*k1)
        k3 = delta_t*oscillator(state + 0.5*k2)
        k4 = delta_t*oscillator(state + k3)
        state += (k1 + 2*k2 + 2*k3 + k4)/6
    return x_array, v_array
#-------------------------------------------------------------------------------------------------



#Harmonic Oscillator function that returns the derivative functions. 
def harmonic_osc(state):
    x = state[0]
    v = state[1]
    x_prime = v
    v_prime = -x #omega = 1.
    return np.array([x_prime,v_prime], float)


#Plotting harmonic oscillator amplitudes:
plt.figure()
plt.title('Simple 1-D Harmonic Oscillator')
plt.xlabel('Time')
plt.ylabel('Position')
plt.plot(t, xv_evolution(1,0,harmonic_osc)[0], color = 'b', label = 'Max amplitude = 1')
plt.plot(t, xv_evolution(2,0,harmonic_osc)[0], color = 'r', label = 'Max amplitude = 2')
plt.legend()
plt.savefig('Harmonic_Oscillator')

#Plotting harmonic oscillator phase space:
plt.figure()
plt.title('1-D Harmonic Oscillator Phase Space')
plt.xlabel('Position')
plt.ylabel('Velocity')
plt.plot(xv_evolution(1,0,harmonic_osc)[0], xv_evolution(1,0,harmonic_osc)[1], color = 'm')
plt.savefig('Harmonic_Oscillator_PhaseSpace')
#-------------------------------------------------------------------------------------------------



#Anharmonic Oscillator function that returns the derivative functions. 
def anharmonic_osc(state):
    x = state[0]
    v = state[1]
    x_prime = v
    v_prime = -x**3 #omega = 1.
    return np.array([x_prime,v_prime], float)


#Plotting anharmonic oscillator amplitudes:
plt.figure()
plt.title('1-D Anharmonic Oscillator')
plt.xlabel('Time')
plt.ylabel('Position')
plt.plot(t, xv_evolution(1,0,anharmonic_osc)[0], color = 'b', label = 'Max amplitude = 1')
plt.plot(t, xv_evolution(2,0,anharmonic_osc)[0], color = 'r', label = 'Max amplitude = 2')
plt.plot(t, xv_evolution(0.5,0,anharmonic_osc)[0], color = 'm', label = 'Max amplitude = 0.5')
plt.legend()
plt.savefig('Anharmonic_Oscillator')
#------------------------------------------------------------------------------------------------



#New time parameters for part e)
t_0 = 0
t_f = 20
N = 10000
delta_t = (t_f - t_0)/N
t = np.arange(t_0, t_f, delta_t)


def Pol_solver(x_0, v_0, mu):
    #van der Pol oscillator function that returns the derivative functions.
    def Pol_osc(state, mu):
        x = state[0]
        v = state[1]
        x_prime = v
        v_prime = mu*(1-x**2)*v - x #omega=1
        return np.array([x_prime,v_prime], float)
    
    x_array = []
    v_array = []
    state = np.array([x_0,v_0], float) #Initializing x,v state array.

    for t_p in t:
        x_array.append(state[0])
        v_array.append(state[1])
        
        #Runge-Kutta 4th Order implimentation:
        k1 = delta_t*Pol_osc(state, mu)
        k2 = delta_t*Pol_osc(state + 0.5*k1, mu)
        k3 = delta_t*Pol_osc(state + 0.5*k2, mu)
        k4 = delta_t*Pol_osc(state + k3, mu)
        state += (k1 + 2*k2 + 2*k3 + k4)/6
    return x_array, v_array


# Plotting Pol oscillator with different mu:
plt.figure()
plt.title('van der Pol Oscillator Phase Space')
plt.xlabel('Position')
plt.ylabel('Velocity')
plt.plot(Pol_solver(1,0,1)[0], Pol_solver(1,0,1)[1], color = 'b', label=r'$\mu = 1$')
plt.plot(Pol_solver(1,0,2)[0], Pol_solver(1,0,2)[1], color = 'g', label=r'$\mu = 2$')
plt.plot(Pol_solver(1,0,4)[0], Pol_solver(1,0,4)[1], color = 'm', label=r'$\mu = 4$')
plt.legend()
plt.savefig('Pol_Oscillator')
#-------------------------------------------------------------------------------------------------


#                                         F I N 