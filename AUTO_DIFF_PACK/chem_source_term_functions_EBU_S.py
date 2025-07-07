import jax.numpy as jnp

#Rate Constants
Ea = 31.588e3 # cal/mol
Ea = Ea*4.184 # J/mol
R = 8.314 # J/molK
Ta = Ea/R # K

a_1 = 1.0
a_2 = 1.0
a = a_1 + a_2


KinP = jnp.array([Ta,a_1,a_2],dtype=jnp.float64)
# nu_p_k = jnp.array([1.0,2.0,0.0,0.0,7.52],dtype=jnp.float64)
# nu_dp_k = jnp.array([0.0,0.0,1.0,2.0,7.52],dtype=jnp.float64)
# nu_k = nu_dp_k - nu_p_k

def w_mol(rho, T, Y0, Y1, Y2, Y3, Y4, C_EBU, k, epsilon,W_k):
    """Compute the w_mol term for the given inputs.

    Args:
        rho (float): Density of the fluid.
        T (float): Temperature of the fluid.
        Y0, Y1, Y2, Y3, Y4): (float): Concentrations of different species.
    
    Expresssion: A*exp(-Ta/T)*(rho*(Y_CH4/W_CH4))**a)*(rho*(Y_O2/W_O2))**b)

    Returns:
        float: Computed w_mol term.
    """
    # Compute w_mol term
    rateConst  = A * jnp.exp(-KinP[0]/T)
    w_mol_term = rateConst * ((rho*(Y0/W_k[0]))**KinP[1]) * ((rho*(Y1/W_k[1]))**KinP[2])
    return w_mol_term

def omega_dot_CH4(rho, T, Y0, Y1, Y2, Y3, Y4, A, k, epsilon, W_k,nu_k):
    # Compute omega_dot for CH4
    omega_dot = nu_k[0]*W_k[0]*w_mol(rho,T,Y0,Y1,Y2,Y3,Y4,A,k,epsilon,W_k)
    return omega_dot

def omega_dot_O2(rho, T, Y0, Y1, Y2, Y3, Y4, A, k, epsilon, W_k, nu_k):
    # Compute omega_dot for O2
    omega_dot = nu_k[1]*W_k[1]*w_mol(rho,T,Y0,Y1,Y2,Y3,Y4,A,k,epsilon,W_k)
    return omega_dot

def omega_dot_CO2(rho, T, Y0, Y1, Y2, Y3, Y4, A, k, epsilon,W_k, nu_k):
    # Compute omega_dot for CO2
    omega_dot = nu_k[2]*W_k[2]*w_mol(rho,T,Y0,Y1,Y2,Y3,Y4,A,k,epsilon,W_k)
    return omega_dot

def omega_dot_H2O(rho, T, Y0, Y1, Y2, Y3, Y4, A, k, epsilon,W_k, nu_k):
    # Compute omega_dot for H2O
    omega_dot = nu_k[3]*W_k[3]*w_mol(rho,T,Y0,Y1,Y2,Y3,Y4,A,k,epsilon,W_k)
    return omega_dot

def omega_dot_N2(rho, T, Y0, Y1, Y2, Y3, Y4, A, k, epsilon,W_k, nu_k):
    # Compute omega_dot for N2
    omega_dot = nu_k[4]*W_k[4]*w_mol(rho,T,Y0,Y1,Y2,Y3,Y4,A,k,epsilon,W_k)
    return omega_dot

def omega_dot_T(rho, T, Y0, Y1, Y2, Y3, Y4, A, k, epsilon, h_f, W_k, nu_k):
    # Compute omega_dot for N2
    wd_CH4 = omega_dot_CH4(rho, T, Y0, Y1, Y2, Y3, Y4, A, k, epsilon,W_k, nu_k)
    wd_O2 = omega_dot_O2(rho, T, Y0, Y1, Y2, Y3, Y4, A, k, epsilon,W_k, nu_k)
    wd_CO2 = omega_dot_CO2(rho, T, Y0, Y1, Y2, Y3, Y4, A, k, epsilon,W_k, nu_k)
    wd_H2O = omega_dot_H2O(rho, T, Y0, Y1, Y2, Y3, Y4, A, k, epsilon,W_k, nu_k)
    wd_N2 = omega_dot_N2(rho, T, Y0, Y1, Y2, Y3, Y4, A, k, epsilon,W_k, nu_k)

    omega_dot_T = -(h_f[0]*wd_CH4 + h_f[1]*wd_O2 + h_f[2]*wd_CO2 + h_f[3]*wd_H2O + h_f[4]*wd_N2)
    return omega_dot_T

def HRR_differentials(w1,w2,w3,w4,w5):
    """Compute the HRR differentials for the given inputs.

    Args:
        w1, w2, w3, w4, w5 (float): Differentials for different species.

    Returns:
        float: Computed HRR differentials.
    """
    # Compute HRR differentials
    hrr_diffs = (w1*h_f1 +
                 w2*h_f2 +
                 w3*h_f3 +
                 w4*h_f4 +
                 w5*h_f5)
    return -1*hrr_diffs