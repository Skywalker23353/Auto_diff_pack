import jax.numpy as jnp
import jax

def C_bar(Y,Y_O2_U,Y_O2_B):
    """Compute the progress variable term for the given species.

    Args:
        Y (float): Concentration of the species.

    Returns:
        float: Progress variable term.
    """
    return (Y_O2_U - Y)/ (Y_O2_U - Y_O2_B)

def domega_mol_drho(Y,C_EBU, epsilon, kappa,Y_O2_U,Y_O2_B):
   
   wmol = C_EBU * epsilon * C_bar(Y,Y_O2_U,Y_O2_B)* (1 - C_bar(Y,Y_O2_U,Y_O2_B)) / kappa
   
   return wmol

def domega_mol_dY2(rho, Y, C_EBU, epsilon, kappa,Y_O2_U, Y_O2_B):
   
   wmol =  C_EBU*(epsilon/kappa)*rho*((Y_O2_U + Y_O2_B - 2*Y)/(Y_O2_U - Y_O2_B)**2)
   
   return wmol

def domega_dot_T_dY2(hf, nu, W, C_EBU, epsilon, kappa, Y_O2_U, Y_O2_B, rho, Y_O2):
   w_T = -(hf[0]*nu[0]*W[0] + hf[1]*nu[1]*W[1] + hf[2]*nu[2]*W[2] + hf[3]*nu[3]*W[3] + hf[4]*nu[4]*W[4]) * domega_mol_dY2(rho, Y_O2, C_EBU, epsilon, kappa, Y_O2_U, Y_O2_B)
   return w_T

def domega_dot_T_drho(hf, nu, W, C_EBU, epsilon, kappa, Y_O2_U, Y_O2_B, rho, Y_O2):
   w_T = -(hf[0]*nu[0]*W[0] + hf[1]*nu[1]*W[1] + hf[2]*nu[2]*W[2] + hf[3]*nu[3]*W[3] + hf[4]*nu[4]*W[4]) * domega_mol_drho(Y_O2, C_EBU, epsilon, kappa, Y_O2_U, Y_O2_B)
   return w_T