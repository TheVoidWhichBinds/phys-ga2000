import numpy as np
import matplotlib.pyplot as plt
from Minimizer import *
from Integrator import ODESolver
from Utilities import *

if __name__ == "__main__":
    # Run the sun test
    num_iter = 1000
    step_size = 1/num_iter
    scale_factors = UnitScalingFactors(M_sun, R_sun, L_sun)
    extra_params = generate_extra_parameters(M_sun, R_sun, L_sun, E_0_sun, kappa_0_sun, mu_sun)

    #print("Extra:", constants)
    optimal_core = run_minimizer(1E16/scale_factors[PRESSURE_UNIT_INDEX],1E7/scale_factors[TEMP_UNIT_INDEX], num_iter, step_size, M_sun, R_sun, L_sun, 
                                       extra_params["E_0_prime"], extra_params["kappa_0_prime"], extra_params["mu"])

    core_initial = gen_core_conditions(optimal_core.x[0], optimal_core.x[1], step_size, extra_params) 
    outer_initial = gen_outer_conditions()
    outwards_sol = ODESolver(core_initial, num_iter, extra_params, False)
    inwards_sol = ODESolver(outer_initial, num_iter, extra_params, True)
    

    state_sun = np.append(outwards_sol[0:num_iter//2, :], np.flipud(inwards_sol[0:num_iter//2, :]), axis=0)
    np.savetxt("SunMesh.txt", state_sun, delimiter=",")
