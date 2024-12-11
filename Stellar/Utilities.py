import numpy as np

# values of fundamental constants in SI units
G = np.float64(6.6743E-11) # (Nm^{2})/(kg^{2})
StefanBoltz = np.float64(5.67E-8) # W/(m^{2}K^{4})
Boltzman = np.float64(1.380649E-23) # J/K
m_p = np.float64(1.67262192E-27) # kg

# Parameters for the Sun
M_sun = 1.989E30 # kg
R_sun  = 6.9634E8 # m
L_sun = 3.8E26 # W
mu_sun = np.float64(0.6)
E_0_sun = 1E-29 # m^5/(kg s^3*K^4)
kappa_0_sun = 1E3 # m^2/kg

# Numerical Resolution used throughout the sim
global_tolerance = 1E-2

MASS_UNIT_INDEX = 0
RADIUS_UNIT_INDEX = 1
DENSITY_UNIT_INDEX = 2
PRESSURE_UNIT_INDEX = 3
LUMINOSITY_UNIT_INDEX = 4
TEMP_UNIT_INDEX = 5
#TIME_UNIT_INDEX = 6 #                             We are currently never using this - Kill?


def UnitScalingFactors(M_0, R_0):
    """
    Returns the scaling factors to convert the unitless 
    numbers in the sim to physical units.
    Inputs:
        M_0: Mass Scale
        R_0: Length Scale
    Output:
        6x1 numpy array whose elements are the scaling factors.
        Use the *_UNIT_INDEX variables to get the corresponding scalings
    """
    assert(R_0 > 0)
    assert(M_0 > 0)
    #The "out" variables are the coefficients which multiply the scaled variables and generate the orignal unit variables.
    M_0 = np.float64(M_0)
    R_0 = np.float64(R_0)
    rho_0 = np.float64(M_0 / (R_0**3))
    t_0 = np.float64(np.sqrt((R_0**3) / (G * M_0)))
    P_0 = np.float64(M_0 / (R_0 * t_0**2))
    L_0 = np.float64(M_0 * R_0**2 / (t_0**3))
    T_0 = np.float64(m_p * R_0**2 / (t_0**2 * Boltzman))

    return np.array([M_0, R_0, rho_0, P_0, L_0, T_0, t_0])



def generate_extra_parameters(M_0, R_0, E_0, kappa_0, mu):
    """
        Given the unitful parameters of the problem, generate the unitless constants to be used in the simulation
    Inputs:
        R_0: Length Scale (maximum radius)
        M_0: Mass Scale (total mass)
        epsilon_0: nuclear energy generation constant for luminosity equation [erg·cm^3/g^2/s] dependent on main fusion reaction (proton-proton in the Sun)
        kappa: opacity parameter for temperature equation [cm^2/g]
        mu: mean molecular weight in units of proton mass
    Output:
        extra_const_params: python dictionary containing the converted constant parameters
    """
    _,_,_,_,_, T_0, t_0 = UnitScalingFactors(M_0, R_0)
    
    mu_prime = mu * m_p * G * M_0 / (Boltzman * R_0 * T_0)
    E_0_prime = E_0 * t_0**3 * M_0 * T_0**4 * np.power(R_0,-5)
    kappa_0_prime = kappa_0 * 3 * M_0**3 / ((16*np.pi)**2 * StefanBoltz * R_0**5 * np.power(T_0,7.5) * np.power(t_0,3))
    extra_params = {"mu_prime": mu_prime, "E_0_prime": E_0_prime, "kappa_0_prime": kappa_0_prime}
    
    return extra_params



def nuclear_energy(rho_prime, T_prime, extra_params):
    """
        generate the nuclear energy production rate given the density and temperature
        Input:
            rho_prime: dimensionless density
            T_prime: dimensionless temperature
        Output:
            E_prime: dimensionless energy rate
    """
    E_prime = (extra_params["E_prime"])*rho_prime**(1)*T_prime**(4)
    return E_prime
