import jax.numpy as jnp
a_1 = 1.0
a_2 = 1.0
a = a_1 + a_2


KinP = jnp.array([a_1,a_2],dtype=jnp.float64)
R = 8.314 # J/molK

def w_mol(rho, T, Y0, Y1, Y2, Y3, Y4, A, Ea, kappa, epsilon, W_k, T_u, T_c, delta):
    """Compute the Q term for the given inputs.

    Args:
        rho (float): Density of the fluid.
        T (float): Temperature of the fluid.
        Y0, Y1, Y2, Y3, Y4): (float): Concentrations of different species.
        A (float): Pre-exponential factor.
        k (float): TKE.
        epsilon (float): Dissipation.
        W_k (array): Molecular weights of the species.
    
    Expresssion: A*exp(-Ta/T)*(rho*(Y_CH4/W_CH4))**a)*(rho*(Y_O2/W_O2))**b)

    Returns:
        float: Computed Q term.
    """
    # Compute Q term
    rateConst  = A * jnp.exp(-Ea/(R*T))
    cold_bc= A * jnp.exp(-Ea/(R*T_u))
    rateConst_cold_bc = rateConst - cold_bc
    w_mol_term = rateConst_cold_bc * ((rho*(Y0/W_k[0]))**KinP[0]) * ((rho*(Y1/W_k[1]))**KinP[1])

    return w_mol_term

def omega_dot_CH4(rho, T, Y0, Y1, Y2, Y3, Y4, A, Ea, kappa, epsilon, W_k, nu_k, T_u, T_c, delta):
    # Compute omega_dot for CH4
    omega_dot = nu_k[0]*W_k[0]*w_mol(rho,T,Y0,Y1,Y2,Y3,Y4,A,Ea,kappa,epsilon,W_k, T_u, T_c, delta)
    return omega_dot

def omega_dot_O2(rho, T, Y0, Y1, Y2, Y3, Y4, A, Ea, kappa, epsilon, W_k, nu_k, T_u, T_c, delta):
    # Compute omega_dot for O2
    omega_dot = nu_k[1]*W_k[1]*w_mol(rho,T,Y0,Y1,Y2,Y3,Y4,A,Ea,kappa,epsilon,W_k, T_u, T_c, delta)
    return omega_dot

def omega_dot_CO2(rho, T, Y0, Y1, Y2, Y3, Y4, A, Ea, kappa, epsilon, W_k, nu_k, T_u, T_c, delta):
    # Compute omega_dot for CO2
    omega_dot = nu_k[2]*W_k[2]*w_mol(rho,T,Y0,Y1,Y2,Y3,Y4,A,Ea,kappa,epsilon,W_k, T_u, T_c, delta)
    return omega_dot

def omega_dot_H2O(rho, T, Y0, Y1, Y2, Y3, Y4, A, Ea, kappa, epsilon, W_k, nu_k, T_u, T_c, delta):
    # Compute omega_dot for H2O
    omega_dot = nu_k[3]*W_k[3]*w_mol(rho,T,Y0,Y1,Y2,Y3,Y4,A,Ea,kappa,epsilon,W_k, T_u, T_c, delta)
    return omega_dot

def omega_dot_N2(rho, T, Y0, Y1, Y2, Y3, Y4, A, Ea, kappa, epsilon, W_k, nu_k, T_u, T_c, delta):
    # Compute omega_dot for N2
    omega_dot = nu_k[4]*W_k[4]*w_mol(rho,T,Y0,Y1,Y2,Y3,Y4,A,Ea,kappa,epsilon,W_k, T_u, T_c, delta)
    return omega_dot

def omega_dot_T(rho, T, Y0, Y1, Y2, Y3, Y4, A, Ea, kappa, epsilon, W_k, nu_k, h_f, T_u, T_c, delta):
    # Compute HRR differentials
    omega_dot_T= -(h_f[0]*omega_dot_CH4(rho, T, Y0, Y1, Y2, Y3, Y4, A, Ea, kappa, epsilon, W_k, nu_k, T_u, T_c, delta) +
                    h_f[1]*omega_dot_O2(rho, T, Y0, Y1, Y2, Y3, Y4, A, Ea, kappa, epsilon, W_k, nu_k, T_u, T_c, delta) +
                    h_f[2]*omega_dot_CO2(rho, T, Y0, Y1, Y2, Y3, Y4, A, Ea, kappa, epsilon, W_k, nu_k, T_u, T_c, delta) +
                    h_f[3]*omega_dot_H2O(rho, T, Y0, Y1, Y2, Y3, Y4, A, Ea, kappa, epsilon, W_k, nu_k, T_u, T_c, delta) +
                    h_f[4]*omega_dot_N2(rho, T, Y0, Y1, Y2, Y3, Y4, A, Ea, kappa, epsilon, W_k, nu_k, T_u, T_c, delta))
    return omega_dot_T
