import jax.numpy as jnp

#Rate Constants

Ea = 31.588e3 # cal/mol
Ea = Ea*4.184 # J/mol
R = 8.314 # J/molK
Ta = Ea/R # K
A = 4.86775e+08
a_1 = 1.0
a_2 = 1.0
a = a_1 + a_2

# Species Data
h_f1 = -5.421277e06 #J/kg
h_f2 = 4.949450e05 #J/kg
h_f3 = -8.956200e06 #J/kg
h_f4 = -1.367883e07 #J/kg
h_f5 = 5.370115e05 #J/kg
hf = [h_f1, h_f2, h_f3, h_f4, h_f5]

W_k = jnp.array([16e-3,32e-3,44e-3,18e-3,28e-3],dtype=jnp.float64)
KinP = jnp.array([Ta,a_1,a_2],dtype=jnp.float64)
nu_p_k = jnp.array([1.0,2.0,0.0,0.0,7.52],dtype=jnp.float64)
nu_dp_k = jnp.array([0.0,0.0,1.0,2.0,7.52],dtype=jnp.float64)
nu_k = nu_dp_k - nu_p_k

def Q(rho, T, Y0, Y1, Y2, Y3, Y4):
    """Compute the Q term for the given inputs.

    Args:
        rho (float): Density of the fluid.
        T (float): Temperature of the fluid.
        Y0, Y1, Y2, Y3, Y4): (float): Concentrations of different species.

    Returns:
        float: Computed Q term.
    """
    # Compute Q term
    rateConst  = A * jnp.exp(-KinP[0]/T)
    Q_term = rateConst * ((rho*(Y0/W_k[0]))**KinP[1]) * ((rho*(Y1/W_k[1]))**KinP[2])
    return Q_term

def omega_dot_CH4(rho, T, Y0, Y1, Y2, Y3, Y4):
    # Compute omega_dot for CH4
    omega_dot = nu_k[0]*W_k[0]*Q(rho,T,Y0,Y1,Y2,Y3,Y4)
    return omega_dot

def omega_dot_O2(rho, T, Y0, Y1, Y2, Y3, Y4):
    # Compute omega_dot for O2
    omega_dot = nu_k[1]*W_k[1]*Q(rho,T,Y0,Y1,Y2,Y3,Y4)
    return omega_dot

def omega_dot_CO2(rho, T, Y0, Y1, Y2, Y3, Y4):
    # Compute omega_dot for CO2
    omega_dot = nu_k[2]*W_k[2]*Q(rho,T,Y0,Y1,Y2,Y3,Y4)
    return omega_dot

def omega_dot_H2O(rho, T, Y0, Y1, Y2, Y3, Y4):
    # Compute omega_dot for H2O
    omega_dot = nu_k[3]*W_k[3]*Q(rho,T,Y0,Y1,Y2,Y3,Y4)
    return omega_dot

def omega_dot_N2(rho, T, Y0, Y1, Y2, Y3, Y4):
    # Compute omega_dot for N2
    omega_dot = nu_k[4]*W_k[4]*Q(rho,T,Y0,Y1,Y2,Y3,Y4)
    return omega_dot

def HRR_differentials(w1,w2,w3,w4,w5):
    """Compute the HRR differentials for the given inputs.

    Args:
        d1, d2, d3, d4, d5 (float): Differentials for different species.

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