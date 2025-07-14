import numpy as np

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
def w(rho, *Y, epsilon, kappa):
    #var structure [Y1,Y2,Y3,Y4,Y5,.......Yn]
    wmol = (epsilon/kappa) * rho * C_bar(Y[1])(1 - C_bar(Y[1]))
    return wmol


def wT_by_C_EBU(rho, T, h_f_all, W_k, nu_k, Y, epsilon, kappa):
    #print("Printing values:") 
    #print(Y.shape)
    #print(Y[1].shape)
    #print(rho.shape)
    #print("Length of Y:", len(Y))
    wT_by_A = np.zeros(rho.shape, dtype=np.float64)
    for k in range(len(Y)):
        wmol = w(rho, *Y, epsilon, kappa)
        wT_by_A += h_f_all[k]*nu_k[k]*W_k[k]*wmol
    wT_by_A = -wT_by_A
    return wT_by_A
