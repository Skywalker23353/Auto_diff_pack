import numpy as np

# omega_dot function
def w(k, rho, T, KinP, W_k, *Y):
    rateConst  = np.exp(-KinP[0]/T)
    wmol = rateConst * ((rho*(Y[0]/W_k[0]))**KinP[1]) * ((rho*(Y[1]/W_k[1]))**KinP[2])
    return wmol


def wT_by_A(rho, T, KinP, h_f_all, W_k, nu_k, Y):
    #print("Printing values:") 
    #print(Y.shape)
    #print(Y[1].shape)
    #print(rho.shape)
    #print("Length of Y:", len(Y))
    wT_by_A = np.zeros(rho.shape, dtype=np.float64)
    for k in range(len(Y)):
        wmol = w(k, rho, T, KinP, W_k, *Y)
        wT_by_A += h_f_all[k]*nu_k[k]*W_k[k]*wmol
    wT_by_A = -wT_by_A
    return wT_by_A
