import numpy as np
import scipy as sp;
from Utilities import *
import copy

np.seterr(all='warn')
# Derivatives of the dependent variables.
# Notably, density is absent. Use Utilities.equation_of_state to generate the density.

def derivative_calc(current, extra_const_params):
    """
        Differential equation for pressure
        current: 1x6 numpy array containing the current values for each variable. See Utilities for indices  of current vector
        extra_const_parameters: dictionary generated by Utilities.gen_extra_parameters. Not used in all dif. eqs
        Output: 1x6 numpy array stating change in each variable
    """
    output = np.zeros(current.shape)
    new_r = (1/(4*np.pi))*(1/(np.power(current[RADIUS_UNIT_INDEX],2)*current[DENSITY_UNIT_INDEX]))
    output[RADIUS_UNIT_INDEX] = new_r

    new_P = (-1/(4*np.pi))* (current[MASS_UNIT_INDEX])/(np.power(current[RADIUS_UNIT_INDEX], 4) )
    output[PRESSURE_UNIT_INDEX] = new_P

    new_L = extra_const_params["E_prime"]* current[DENSITY_UNIT_INDEX]*np.power(current[TEMP_UNIT_INDEX],4)
    output[LUMINOSITY_UNIT_INDEX] = new_L

    # assume that kappa_prime is in args
    cur_t = current[TEMP_UNIT_INDEX]
    multiplied_vars = cur_t* current[RADIUS_UNIT_INDEX]
    var = np.power(multiplied_vars, -4)
    tp = np.power(cur_t,-2.5) #!!!                   OVERFLOW/NaN being encountered               !!!
    new_T = - extra_const_params["kappa_prime"]* current[DENSITY_UNIT_INDEX] * current[LUMINOSITY_UNIT_INDEX] * var * tp 
    output[TEMP_UNIT_INDEX] = new_T
    output[MASS_UNIT_INDEX] = 1
    output[DENSITY_UNIT_INDEX] = 0
    return output


def RK4(f, current, step_size, extra_const_params):
    """
    Straightforward implementation of RK4 algorithm
    Inputs:
        f: derivative of dependent variable. Takes form f(current, extra_const_params),
            * current encodes the current state of the system as a 6x1 np array
            * constant_dict is a Python dictionary which holds any constants needed amongst all the diff eqs
            f should output a numpy array with size of current (6x1)
        current: numpy array which holds the current state of the system
            0th term is independent variable, and the rest are the dependent ones
        step_size: how big of a step in x do you want
    Output:
        1x6 numpy array containing Derivatives of all variables of current. Independent variable (mass) is just step size. Density is calculated outside of RK4
    """
    assert(step_size >0)
    step_size = np.float64(step_size)
    # half_step_size = step_size/2                    Unnecessary variable assignment.

    # k1 = f(current,extra_const_params)              RK4 Formula incorrect, see pg. 336 of Newman
    # new_input = current + half_step_size*k1 
    # k2 = f(new_input,extra_const_params)
    # new_input =current + half_step_size*k2 
    # k3 = f(new_input,extra_const_params)
    # new_input = current+step_size*k3
    # k4 = f(new_input,extra_const_params)
    # update = (step_size/6) * (k1+2*k2+2*k3+k4)

    dependent_array = np.array([0,1,1,1,1,1])
    mass_array = np.array([1,0,0,0,0,0])
    k1 = step_size * f(current, extra_const_params)
    k2 = step_size * f(current + (k1/2)*dependent_array + (step_size/2)*mass_array, extra_const_params)
    k3 = step_size * f(current + (k2/2)*dependent_array + (step_size/2)*mass_array, extra_const_params)
    k4 = step_size * f(current + k3*dependent_array + step_size*mass_array, extra_const_params)
    update = current + (k1+2*k2+2*k3+k4)/6
    return update


if __name__ == "__main__":
    pass


#Iterates state of system thru RK4, creating an array of the key variables at each mass step.
def ODESolver(initial_conditions, num_steps, extra_const_parameters, verbose=False):
    """
        Inputs:
            initial_conditions (1x6 np array): The initial conditions of the system
            num_steps (np.float64): The number of steps to take between 0 and 1
        Outputs:
            state_matrix ((num_steps x 6) np array):
                The state of the system at each mass step in time. Needed for plotting reasons
                The final state is given by state_matrix[:,-1]
    """
    step_size = 1/num_steps
    cur_state = initial_conditions
    output = [initial_conditions]
    for i in range(1, num_steps):
        #RK4 receives a specific differential equation corresponding to each variable of interest. 
        #RK4 outputs a 6x1 array with elements of: mass, radius, pressure, luminosity, 
        #temperature, and density. Only the element corresponding to the differential equation (derivatives[n])
        #input into RK4 has the correct updated value
        update  = RK4(derivative_calc, cur_state, step_size, extra_const_parameters)
        #
        if(verbose):
            print(i, cur_state, update)
        if(np.any(np.isnan(update))):
            break
        
        #cur_state = cur_state + delta #                naming this "delta" is misleading - RK4 outputs the new state, not the delta between states - renamed instances of "delta" to "update"
        cur_state = update
        cur_state[DENSITY_UNIT_INDEX] = equation_of_state(cur_state[PRESSURE_UNIT_INDEX], cur_state[TEMP_UNIT_INDEX], extra_const_parameters)
        
        if(np.any(np.less(cur_state,0)) or (np.any(np.isnan(cur_state)))):
            break
        output.append( copy.deepcopy(cur_state) )
    #   print("End STEP: ", cur_state)
    o = np.vstack(output)
    return o

if __name__ == "__main__":
    pass