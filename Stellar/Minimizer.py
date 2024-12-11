import numpy as np
import scipy as sp
import Integrator
from Utilities import *

#MASS_UNIT_INDEX = 0 []
#RADIUS_UNIT_INDEX = 1 []
#DENSITY_UNIT_INDEX = 2 []
#PRESSURE_UNIT_INDEX = 3 [dynes/cm^2]
#LUMINOSITY_UNIT_INDEX = 4 []
#TEMP_UNIT_INDEX = 5 [K]


def gen_core_guess(P_core, T_core, extra_params):
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
    rho_core = equation_of_state(P_core, T_core, extra_params)
    M_core = global_tolerance
    R_core = global_tolerance
    L_core = global_tolerance
    core_conds = np.array([M_core, R_core, rho_core, P_core, L_core, T_core])
    return core_conds



def gen_outer_guess(L_outer):
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
    rho_outer = global_tolerance
    M_outer = 1
    R_outer = 1
    P_outer = global_tolerance
    T_outer = global_tolerance
    outer_conds = np.array((M_outer, R_outer, rho_outer, P_outer, L_outer, T_outer))
    return outer_conds




def smooth_merge(bound_guess, num_iter, extra_params, step_size):
    """
        Input:
            estimator_guess: 2x1 numpy array of the form [temp, pressure]. These should be unitless
            *args: Expect two additional args: the ODE solver, and the number of steps the ODE solver should take
        Output:
            np.float64: the loss function for the given initial pressure and temp conditions
    """
    core_guess = np.array(bound_guess[:6])
    outer_guess = np.array(bound_guess[6:])
    outwards_sol,_,outwards_deriv = Integrator.ODESolver(core_guess, num_iter, extra_params, False)
    inwards_sol,inwards_deriv,_ = Integrator.ODESolver(outer_guess, num_iter, extra_params, True)
    #
    if outwards_deriv is None:
        outwards_deriv = np.zeros((1, 6))
    if inwards_deriv is None:
        inwards_deriv = np.zeros((1, 6))
    #
    deriv_diff = np.sum(np.abs(outwards_deriv - inwards_deriv))
    deriv_weight = 1
    func_diff = np.sum(np.abs(outwards_sol[num_iter//2,:] - inwards_sol[num_iter//2,:]))
    func_weight = 100

    return deriv_weight * deriv_diff**2 + func_weight * func_diff**2




def run_minimizer(core_guess, outer_guess, num_iters, step_size, M_0, R_0, E_0, kappa_0, mu):
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
    extra_params = generate_extra_parameters(M_0, R_0, E_0, kappa_0, mu)
    scaling_factors = UnitScalingFactors(M_0, R_0)[0:6]
    bound_guess = np.hstack((core_guess/scaling_factors, outer_guess/scaling_factors))

    strict = (
    {'type': 'ineq', 'fun': lambda x: x[0] - 1E-9}, #All elements of x must be positive
    {'type': 'ineq', 'fun': lambda x: x[1] - 1E-9}, #All elements of x must be positive
    {'type': 'ineq', 'fun': lambda x: x[2] - 1E-9}, #All elements of x must be positive
    {'type': 'ineq', 'fun': lambda x: x[3] - 1E-9}, #All elements of x must be positive
    {'type': 'ineq', 'fun': lambda x: x[4] - 1E-9}, #All elements of x must be positive
    {'type': 'ineq', 'fun': lambda x: x[5] - 1E-9}, #All elements of x must be positive
    {'type': 'ineq', 'fun': lambda x: x[6] - 1E-9}, #All elements of x must be positive
    {'type': 'ineq', 'fun': lambda x: x[7] - 1E-9}, #All elements of x must be positive
    {'type': 'ineq', 'fun': lambda x: x[8] - 1E-9}, #All elements of x must be positive
    {'type': 'ineq', 'fun': lambda x: x[9] - 1E-9}, #All elements of x must be positive
    {'type': 'ineq', 'fun': lambda x: x[10] - 1E-9}, #All elements of x must be positive
    {'type': 'ineq', 'fun': lambda x: x[11] - 1E-9}, #All elements of x must be positive
    {'type': 'eq', 'fun': lambda x: x[6] - 1}, #Outer mass starts at 1
    {'type': 'eq', 'fun': lambda x: x[7] - 1} #Outer radius starts at 1
    
                )

    return sp.optimize.minimize(smooth_merge, bound_guess,
                                    args=(num_iters, extra_params, step_size), 
                                    bounds=sp.optimize.Bounds(lb = global_tolerance * np.ones(12), 
                                    ub = np.inf * np.ones(12), keep_feasible = True * np.ones(12)), 
                                    constraints = strict
                                )
    

if __name__ == "__main__":
    pass
