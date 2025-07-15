import jax.numpy as jnp
import jax

#Rate Constants
R = 8.314 # J/molK
#Species Data
h_f1 = -5.421277e06 #J/kg
h_f2 = 4.949450e05 #J/kg
h_f3 = -8.956200e06 #J/kg
h_f4 = -1.367883e07 #J/kg
h_f5 = 5.370115e05 #J/kg
hf = [h_f1, h_f2, h_f3, h_f4, h_f5]
W_k = jnp.array([16e-3,32e-3,44e-3,18e-3,28e-3],dtype=jnp.float64)
nu_p_k = jnp.array([1.0,2.0,0.0,0.0,7.52],dtype=jnp.float64)
nu_dp_k = jnp.array([0.0,0.0,1.0,2.0,7.52],dtype=jnp.float64)
nu_k = nu_dp_k - nu_p_k

Y_O2_B = 0.0423
Y_O2_U = 0.2226

def C_bar(Y):
    """Compute the progress variable term for the given species.

    Args:
        Y (float): Concentration of the species.

    Returns:
        float: Progress variable term.
    """
    return (Y_O2_U - Y)/ (Y_O2_U - Y_O2_B)

# omega_dot function
def w(C_EBU, k, rho, T, *Y, epsilon, kappa):
    #var structure [Y1,Y2,Y3,Y4,Y5,.......Yn]
    rateConst  = C_EBU
    wmol = rateConst * epsilon * rho * C_bar(Y[1])(1 - C_bar(Y[1])) / kappa
    return wmol

# -------------Compute Derivatives-------------------
def return_wT_deriv(C_EBU,var_idx, rho, T, *Y, epsilon, kappa):
     #Funtion inputs: var_idx, rho, T, Y1, Y2, Y3, Y4, Y5,....,Yn
     #var_idx : To specify the variable with respect to which the derivative is to be computed
     #rho, T : Density and Temperature
     #Y : Species mass fractions
     print("Computing wT deriv for variable index = ",var_idx)
     wT_deriv = jnp.zeros(len(rho), dtype=jnp.float64)
     wk_deriv = jax.jacfwd(w,var_idx)
    #  dw_dvaridx_at_qbar = dw_dq(rho, T, *Y) # Jacobian of w wrt q at qbar
    #  dw_dvaridx_at_qbar = jnp.diag(dw_dvaridx_at_qbar) #picking only the diagonal elements
     for k in range(len(Y)):
        wk_deriv_at_qbar = wk_deriv(C_EBU, k, rho, T, *Y, epsilon, kappa) # Jacobian of w wrt q at qbar
        wk_deriv_at_qbar = jnp.diag(wk_deriv_at_qbar) #picking only the diagonal elements
        temp = hf[k]*nu_k[k]*W_k[k]*wk_deriv_at_qbar
        wT_deriv += temp
        del temp 
     wT_deriv = -wT_deriv
     return wT_deriv
# ---------------------------------------------------
# def return_omega_dot_k(A, k, q, rho, T, *Y):
#     print("Computing omega_dot_k for species = ",k)
#     omega_dot_k = jnp.zeros(len(rho), dtype=jnp.float64)
#     wk_deriv = jax.jacfwd(w,q)
#     wk_deriv_at_qbar = wk_deriv(A, k, rho, T, *Y)
#     wk_deriv_at_qbar = jnp.diag(wk_deriv_at_qbar) #picking only the diagonal elements
#     omega_dot_k = nu_k[k]*W_k[k]*wk_deriv_at_qbar
#     return omega_dot_k
# ---------------------------------------------------
def domega_dot_drho_actual_deriv(rho, T, Y1, Y2, Y3, Y4, C_EBU, epsilon, kappa):
   
   rateConst  = C_EBU
   wmol = rateConst * epsilon * C_bar(Y2)* (1 - C_bar(Y2)) / kappa
   return wmol

def domega_dot_dY2_actual_deriv(rho, T, Y1, Y2, Y3, Y4, C_EBU, epsilon, kappa,Y_O2_U, Y_O2_B):
   
   wmol =  C_EBU*(epsilon/kappa)*rho*((Y_O2_U + Y_O2_B - 2*Y2)/(Y_O2_U - Y_O2_B)**2) # No dependence on Y1 in this formulation
   
   return wmol
