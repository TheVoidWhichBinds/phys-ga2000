import numpy as np
import scipy as sp
import Integrator
import Utilities

#MASS_UNIT_INDEX = 0 []
#RADIUS_UNIT_INDEX = 1 []
#DENSITY_UNIT_INDEX = 2 []
#PRESSURE_UNIT_INDEX = 3 [dynes/cm^2]
#LUMINOSITY_UNIT_INDEX = 4 []
#TEMP_UNIT_INDEX = 5 [K]


def gen_core_conditions(P_core, T_core, step_size, extra_params):
    """
        Input:
            Helper function to deal with the fact that we can't start at m=0. We pretend that the central density is roughly constant, then fudge the boundary conditions a bit
            starting_scaled_temp: the initial scaled temperature at the center of the star (unitless)
            starting_scaled_pressure: same as temp, but for pressure
            step_size: the mass step size to be taken (typically, this should be half a step size of your actual simulation)
            const_params: dictionary containing the constant parameters of the problem. Generated from Utilities.generate_extra_parameters
        Output:
            1x6 numpy array containing initial conditions in scaled variables
    """
    # We need to fudge the radius initial condition to avoid the singularity at r=0 in the equations.
    rho_core =  Utilities.equation_of_state(P_core, T_core, extra_params)
    M_core = step_size/10
    R_core = np.power((4*np.pi/3)*M_core/rho_core, 1/3)
    L_core = 0
    core_conds = np.array([M_core, R_core, rho_core, P_core, L_core, T_core])
    return core_conds



def gen_outer_conditions():
    """
        Input:
            Helper function to deal with the fact that we can't start at m=0. We pretend that the central density is roughly constant, then fudge the boundary conditions a bit
            starting_scaled_temp: the initial scaled temperature at the center of the star (unitless)
            starting_scaled_pressure: same as temp, but for pressure
            step_size: the mass step size to be taken (typically, this should be half a step size of your actual simulation)
            const_params: dictionary containing the constant parameters of the problem. Generated from Utilities.generate_extra_parameters
        Output:
            1x6 numpy array containing initial conditions in scaled variables
    """
    rho_outer = Utilities.global_tolerance
    M_outer = 1
    R_outer = 1
    L_outer = 1
    P_outer = Utilities.global_tolerance
    T_outer = Utilities.global_tolerance
    outer_conds = np.array((M_outer, R_outer, rho_outer, P_outer, L_outer, T_outer))
    return outer_conds




def halfway_diff(core_guess, num_iter, extra_params, step_size):
    """
        Input:
            estimator_guess: 2x1 numpy array of the form [temp, pressure]. These should be unitless
            *args: Expect two additional args: the ODE solver, and the number of steps the ODE solver should take
        Output:
            np.float64: the loss function for the given initial pressure and temp conditions
    """
# We assume that estimator has dimensions 2x1
# Temperature and pressure normally can't be negative
    P_guess = core_guess[0]
    T_guess = core_guess[1]
    assert(P_guess >= 0) 
    assert(T_guess >= 0)
    
    outwards = Integrator.ODESolver(gen_core_conditions(P_guess, T_guess, step_size, extra_params), num_iter, extra_params, False)
    inwards =  Integrator.ODESolver(gen_outer_conditions(), num_iter, extra_params, True)
    diff = np.sum(outwards[num_iter//2,:] - inwards[num_iter//2,:])
    diff_weight = 1
    
    core_conditions_solved = outwards[0,[0,1,4]]
    outer_conditions_solved = inwards[0,:]
    boundary = np.sum(np.abs(Utilities.global_tolerance*np.array([1,1,1]) - core_conditions_solved))  +  np.sum(np.abs(np.array([1,1,0,0,1,0]) - outer_conditions_solved))
    boundary_weight = 0
    
    return diff_weight*diff**2 + boundary_weight*boundary



def run_minimizer(P_guess, T_guess, num_iters, step_size, M_0, R_0, L_0, E_0, kappa, mu):
    """
        Helper function: to generate set up the minimizer and run it
            Input:
                Initial_scaled_T: Initial guess of temperature (unitless)
            Initial_scaled_P: Initial guess of pressure (unitless)
            num_iters: how many steps the integrator should take (int >0)
            M_0: the relevant mass scale of the problem (kg)
            R_0: The relevant distance scale of the problem (m)
            epsilon: the e_0 parameter in the luminosity differential equation
            kappa: the k_0 parameter in the temperature differential equation
            mu: the mean molecular weight in units of proton mass
        Output:
            OptimizeResult from scipy.optimize.minimize
    """
    core_guess = np.array([P_guess, T_guess])
    extra_params = Utilities.generate_extra_parameters(M_0, R_0, L_0, E_0, kappa, mu)

    return sp.optimize.minimize(halfway_diff, core_guess,
                                    args=(num_iters, extra_params, step_size),  # no need for 'solver' here unless it's used inside 'halfway_diff'
                                    bounds=sp.optimize.Bounds(lb=[Utilities.global_tolerance, Utilities.global_tolerance], 
                                    ub=[np.inf, np.inf], keep_feasible=[True, True]),)
    

if __name__ == "__main__":
    pass
