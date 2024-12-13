import numpy as np



import numpy as np

def f_r(r_u, r_k, m_u, m_k, rho_k, rho_u):
    return (r_u - r_k) / (m_u - m_k)  -  (2/np.pi) / ((rho_k + rho_u) * (r_k + r_u)**2)

def f_p(p_u, p_k, m_u, m_k, r_k, r_u, rho_k, rho_u):
    return (p_u - p_k) / (m_u - m_k)  +  (2 / np.pi) * (m_k + m_u) / ((r_k + r_u)**4) 

def f_L(L_u, L_k, m_u, m_k, rho_k, rho_u, epsilon_0, T_k, T_u):
    return (L_u - L_k) / (m_u - m_k)  -  (epsilon_0 / 2**5) * (rho_k + rho_u) * (T_k + T_u)**4

def f_T(T_u, T_k, m_u, m_k, rho_k, rho_u, L_k, L_u, kappa_0, r_k, r_u):
    return (T_u - T_k) / (m_u - m_k)  +  (2**8.5 * kappa_0 * (rho_k + rho_u) * (L_k + L_u)) / ((T_k + T_u)**6.5 * (r_k + r_u)**4)
