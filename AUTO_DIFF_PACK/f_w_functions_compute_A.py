import jax.numpy as jnp
import jax

#Rate Constants

Ea = 31.588e3 # cal/mol
Ea = Ea*4.184 # J/mol
R = 8.314 # J/molK
Ta = Ea/R # K
a_1 = 1.0
a_2 = 1.0
a = a_1 + a_2
KinP = jnp.array([Ta,a_1,a_2],dtype=jnp.float64)


#Species Data
h_f1 = -5.421277e06 #J/kg
h_f2 = 4.949450e05 #J/kg
h_f3 = -8.956200e06 #J/kg
h_f4 = -1.367883e07 #J/kg
h_f5 = 5.370115e05 #J/kg
hf_actual = [h_f1, h_f2, h_f3, h_f4, h_f5]
#scalefac = -1/h_f4
scalefac = 1
h_f1 = h_f1 * scalefac #J/kg
h_f2 = h_f2 * scalefac#J/kg
h_f3 = h_f3 * scalefac#J/kg
h_f4 = h_f4 * scalefac#J/kg
h_f5 = h_f5 * scalefac#J/kg
hf = [h_f1, h_f2, h_f3, h_f4, h_f5]
W_k = jnp.array([16e-3,32e-3,44e-3,18e-3,28e-3],dtype=jnp.float64)
nu_p_k = jnp.array([1.0,2.0,0.0,0.0,7.52],dtype=jnp.float64)
nu_dp_k = jnp.array([0.0,0.0,1.0,2.0,7.52],dtype=jnp.float64)
nu_k = nu_dp_k - nu_p_k

# omega_dot function
def w(k, rho, T, *Y):
    #var structure [Y1,Y2,Y3,Y4,Y5,.......Yn]
    rateConst  = jnp.exp(-KinP[0]/T)
    wmol = rateConst * ((rho*(Y[0]/W_k[0]))**KinP[1]) * ((rho*(Y[1]/W_k[1]))**KinP[2])
    return wmol


def wT(A, rho, T, *Y):
    wT = jnp.zeros(rho.shape, dtype=jnp.float64)
    for k in range(len(Y)):
        wmol = w(A, k, rho, T, *Y)
        wT += hf_actual[k]*nu_k[k]*W_k[k]*wmol
    wT = -wT
    return wT
