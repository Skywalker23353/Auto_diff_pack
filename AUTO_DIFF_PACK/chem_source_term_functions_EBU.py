#Rate Constants
import jax.numpy as jnp
R = 8.314 # J/molK
tolerance = 1e-3

def w_mol(rho, T, Y0, Y1, Y2, Y3, Y4, C_EBU, kappa, epsilon, W_k, Y_O2_U_vec, Y_O2_B_vec):
    """Compute the Q term for the given inputs.

    Args:
        rho (float): Density of the fluid.
        T (float): Temperature of the fluid.
        Y0, Y1, Y2, Y3, Y4): (float): Concentrations of different species.
        C_EBU (float): EBU factor.
        k (float): TKE.
        epsilon (float): Dissipation.
        W_k (array): Molecular weights of the species.
    
    Expresssion: A*exp(-Ta/T)*(rho*(Y_CH4/W_CH4))**a)*(rho*(Y_O2/W_O2))**b)

    Returns:
        float: Computed Q term.
    """
    C = (Y_O2_U_vec - Y1)/(Y_O2_U_vec - Y_O2_B_vec)
    C = jnp.where(C < tolerance, 0.0, C)
    # Compute Q term
    rateConst  = C_EBU
    w_mol_term = rateConst * (epsilon/kappa) * rho * C * (1-C)
    return w_mol_term

def omega_dot_CH4(rho, T, Y0, Y1, Y2, Y3, Y4, C_EBU,kappa, epsilon, W_k, nu_k, Y_O2_U_vec, Y_O2_B_vec):
    # Compute omega_dot for CH4
    omega_dot = nu_k[0]*W_k[0]*w_mol(rho,T,Y0,Y1,Y2,Y3,Y4,C_EBU,kappa,epsilon,W_k, Y_O2_U_vec, Y_O2_B_vec)
    return omega_dot

def omega_dot_O2(rho, T, Y0, Y1, Y2, Y3, Y4, C_EBU, kappa, epsilon, W_k, nu_k, Y_O2_U_vec, Y_O2_B_vec):
    # Compute omega_dot for O2
    omega_dot = nu_k[1]*W_k[1]*w_mol(rho,T,Y0,Y1,Y2,Y3,Y4,C_EBU,kappa,epsilon,W_k, Y_O2_U_vec, Y_O2_B_vec)
    return omega_dot

def omega_dot_CO2(rho, T, Y0, Y1, Y2, Y3, Y4, C_EBU, kappa, epsilon, W_k, nu_k, Y_O2_U_vec, Y_O2_B_vec):
    # Compute omega_dot for CO2
    omega_dot = nu_k[2]*W_k[2]*w_mol(rho,T,Y0,Y1,Y2,Y3,Y4,C_EBU,kappa,epsilon,W_k, Y_O2_U_vec, Y_O2_B_vec)
    return omega_dot

def omega_dot_H2O(rho, T, Y0, Y1, Y2, Y3, Y4, C_EBU, kappa, epsilon ,W_k, nu_k, Y_O2_U_vec, Y_O2_B_vec):
    # Compute omega_dot for H2O
    omega_dot = nu_k[3]*W_k[3]*w_mol(rho,T,Y0,Y1,Y2,Y3,Y4,C_EBU,kappa,epsilon,W_k, Y_O2_U_vec, Y_O2_B_vec)
    return omega_dot

def omega_dot_N2(rho, T, Y0, Y1, Y2, Y3, Y4, C_EBU, kappa,epsilon ,W_k, nu_k, Y_O2_U_vec, Y_O2_B_vec):
    # Compute omega_dot for N2
    omega_dot = nu_k[4]*W_k[4]*w_mol(rho,T,Y0,Y1,Y2,Y3,Y4,C_EBU,kappa,epsilon,W_k, Y_O2_U_vec, Y_O2_B_vec)
    return omega_dot

def omega_dot_T(rho, T, Y0, Y1, Y2, Y3, Y4, C_EBU,kappa ,epsilon ,W_k, nu_k, h_f, Y_O2_U_vec, Y_O2_B_vec):
    # Compute HRR differentials
    omega_dot_T= -(h_f[0]*omega_dot_CH4(rho, T, Y0, Y1, Y2, Y3, Y4, C_EBU, kappa, epsilon, W_k, nu_k, Y_O2_U_vec, Y_O2_B_vec) +
                    h_f[1]*omega_dot_O2(rho, T, Y0, Y1, Y2, Y3, Y4, C_EBU, kappa, epsilon, W_k, nu_k, Y_O2_U_vec, Y_O2_B_vec) +
                    h_f[2]*omega_dot_CO2(rho, T, Y0, Y1, Y2, Y3, Y4, C_EBU, kappa, epsilon, W_k, nu_k, Y_O2_U_vec, Y_O2_B_vec) +
                    h_f[3]*omega_dot_H2O(rho, T, Y0, Y1, Y2, Y3, Y4, C_EBU, kappa, epsilon, W_k, nu_k, Y_O2_U_vec, Y_O2_B_vec) +
                    h_f[4]*omega_dot_N2(rho, T, Y0, Y1, Y2, Y3, Y4, C_EBU, kappa, epsilon, W_k, nu_k, Y_O2_U_vec, Y_O2_B_vec))
    return omega_dot_T
