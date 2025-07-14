import numpy as np
import jax.numpy as jnp

Y_O2_B = jnp.array([0.0423])
Y_O2_U = jnp.array([0.2226])

def C_bar(Y):
    """Compute the progress variable term for the given species.

    Args:
        Y (float): Concentration of the species.

    Returns:
        float: Progress variable term.
    """
    return (Y_O2_U - Y)/ (Y_O2_U - Y_O2_B)

# omega_dot function
def w(rho, C, epsilon, kappa):
<<<<<<< HEAD
    wmol = (epsilon/kappa) * rho * C * (1 - C)
=======
    wmol = (epsilon/kappa) * rho * C*(1-C)
>>>>>>> e16ad9677e11ff2165852277f144c7b3511b364e
    return wmol


def wT_by_C_EBU(rho, h_f_all, W_k, nu_k, Y0,Y1,Y2,Y3,Y4, epsilon, kappa):
    #print("Printing values:") 
    #print(Y.shape)
    #print(Y[1].shape)
    #print(rho.shape)
    #print("Length of Y:", len(Y))
    C = C_bar(Y1)
    wT_by_A = np.zeros(rho.shape, dtype=np.float64)
    C = C_bar(Y1)
    for k in range(len(nu_k)):
        wmol = w(rho, C, epsilon, kappa)
        wT_by_A += h_f_all[k]*nu_k[k]*W_k[k]*wmol
    wT_by_A = -wT_by_A
    return wT_by_A
