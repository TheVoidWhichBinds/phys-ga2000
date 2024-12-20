import numpy as np
import matplotlib.pyplot as plt
from Minimizer import *
from Integrator import ODESolver
from Utilities import *

if __name__ == "__main__":
    # Run the sun test
    num_iter = 1000
    step_size = 1/num_iter
    scale_factors = UnitScalingFactors(M_sun, R_sun)
    extra_params = generate_extra_parameters(M_sun, R_sun, E_0_sun, kappa_0_sun, mu_sun)

    core_guess = gen_core_guess(1E16, 1E7, extra_params)
    outer_guess = gen_outer_guess(R_sun,L_sun)

    # optimal_init = run_minimizer(core_guess, outer_guess, num_iter, step_size, M_sun, R_sun, 
    #                                 E_0_sun, kappa_0_sun, mu_sun)
    # core_opt = optimal_init.x[:6]
    # outer_opt = optimal_init.x[6:]
    
    # outwards_sol,_,_ = ODESolver(core_opt, num_iter, extra_params, False)
    # inwards_sol,_,_ = ODESolver(outer_opt, num_iter, extra_params, True)
    # state_sun = outwards_sol[2:,:]
    
    outwards_sol,_,_ = ODESolver(core_guess, num_iter, extra_params, False)
    inwards_sol,_,_ = ODESolver(outer_guess, num_iter, extra_params, True)
    state_sun = np.append(outwards_sol[2:,:], inwards_sol[2:,:], axis = 0)

    
    np.savetxt("SunMesh.txt", state_sun, delimiter=",")
