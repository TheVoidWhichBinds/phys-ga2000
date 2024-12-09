import numpy as np
import scipy as sp;
from Utilities import *
import copy
from math import log

np.seterr(all='warn')
# Derivatives of the dependent variables.
# Notably, density is absent. Use Utilities.equation_of_state to generate the density.

def derivative_calc(current, extra_params):
    """
        Differential equation for pressure
        current: 1x6 numpy array containing the current values for each variable. See Utilities for indices  of current vector
        extra_const_parameters: dictionary generated by Utilities.gen_extra_parameters. Not used in all dif. eqs
        Output: 1x6 numpy array stating change in each variable
    """
    output = np.zeros(current.shape)
    dr_dm = (1/(4*np.pi)) * np.power(current[RADIUS_UNIT_INDEX],-2) * np.power(current[DENSITY_UNIT_INDEX],-1)
    output[RADIUS_UNIT_INDEX] = dr_dm

    dP_dm = (-1/(4*np.pi)) * current[MASS_UNIT_INDEX] * np.power(current[RADIUS_UNIT_INDEX],-4)
    output[PRESSURE_UNIT_INDEX] = dP_dm

    dL_dm = extra_params["E_0_prime"] * current[DENSITY_UNIT_INDEX] * np.power(current[TEMP_UNIT_INDEX],4)
    output[LUMINOSITY_UNIT_INDEX] = dL_dm
    
    dT_dm = - extra_params["kappa_0_prime"] * current[LUMINOSITY_UNIT_INDEX] * current[DENSITY_UNIT_INDEX] * np.power(current[RADIUS_UNIT_INDEX],-4) * np.power(current[TEMP_UNIT_INDEX],-6.5)
    output[TEMP_UNIT_INDEX] = dT_dm
    
    # if np.isnan(dT_dm):
    #     print("Warning: dT/dm is NaN.")
    # elif dT_dm == np.inf:
    #     print("Warning: dT/dm overflowed (positive infinity).")
    # elif dT_dm == -np.inf:
    #     print("Warning: dT/dm overflowed (negative infinity).")
    # elif abs(dT_dm) < np.finfo(np.float64).tiny:  # Check for underflow (close to zero)
    #     print(f"Warning: dT/dm underflowed (value: {dT_dm}).")
    
    #print(dT_dm)
    #print(dL_dm)
    return output



def RK4(f, current, step_size, extra_const_params, inwards):
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
    rev = -1 if inwards else 1
    dependent_array = np.array([0,1,1,1,1,1])
    mass_array = np.array([1,0,0,0,0,0])
    
    k1 = step_size * rev * f(current, extra_const_params)
    k2 = step_size * rev * f(current + (k1/2)*dependent_array + rev*(step_size/2)*mass_array, extra_const_params)
    k3 = step_size * rev * f(current + (k2/2)*dependent_array + rev*(step_size/2)*mass_array, extra_const_params)
    k4 = step_size * rev * f(current + k3*dependent_array + rev*step_size*mass_array, extra_const_params)
    update = current + (1/6)*(k1+2*k2+2*k3+k4) * dependent_array + rev * step_size * mass_array
   
    #print(current[LUMINOSITY_UNIT_INDEX])
    #print(current[TEMP_UNIT_INDEX])
    
    return update




if __name__ == "__main__":
    pass

#Iterates state of system thru RK4, creating an array of the key variables at each mass step.
def ODESolver(initial_conditions, num_iter, extra_params, inwards, verbose=False):
    """
        Inputs:
            initial_conditions (1x6 np array): The initial conditions of the system
            num_steps (np.float64): The number of steps to take between 0 and 1
        Outputs:
            state_matrix ((num_steps x 6) np array):
                The state of the system at each mass step in time. Needed for plotting reasons
                The final state is given by state_matrix[:,-1]
    """
    step_size = 1/num_iter #Step size = unitless scale/num_iter
    current = initial_conditions 
    output = np.zeros((num_iter, 6))
    output[0,:] = current
    
    outwards_deriv = np.zeros((1,6))
    inwards_deriv = np.zeros((1,6))

    for i in range(1, num_iter):
        #RK4 receives a specific differential equation corresponding to each variable of interest. 
        #RK4 outputs a 6x1 array with elements of: mass, radius, pressure, luminosity, 
        #temperature, and density. Only the element corresponding to the differential equation (derivatives[n])
        #input into RK4 has the correct updated value
        
        if i == num_iter//2:
            if inwards:
                inwards_deriv = derivative_calc(current, extra_params)
            else:
                outwards_deriv = derivative_calc(current, extra_params)
            
        update  = RK4(derivative_calc, current, step_size, extra_params, inwards)
        
        if np.any(np.isnan(update)):
            if verbose:
                print(f"Iteration {i}: NaN encountered in update, stopping integration.")
            break
        if np.any(np.less(current, 0)) or np.any(np.isnan(current)):
            if verbose:
                print(f"Invalid state encountered: {current}")
            return output, None, None

        
        current = update
        current[DENSITY_UNIT_INDEX] = equation_of_state(current[PRESSURE_UNIT_INDEX], current[TEMP_UNIT_INDEX], extra_params)
        if(np.any(np.less(current,0)) or (np.any(np.isnan(current)))):
             return output, None, None
        output[i,:] = current
    #print(output)
    return output, inwards_deriv, outwards_deriv


if __name__ == "__main__": #                            Purpose of having this at the end?
    pass
