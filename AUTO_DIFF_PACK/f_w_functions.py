import jax.numpy as jnp
import jax

#Rate Constants

Ea = 31.588e3 # cal/mol
Ea = Ea*4.184 # J/mol
R = 8.314 # J/molK
# A = 4.86775e+08
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
hf = jnp.array([h_f1, h_f2, h_f3, h_f4, h_f5])
W_k = jnp.array([16e-3,32e-3,44e-3,18e-3,28e-3],dtype=jnp.float64)
nu_p_k = jnp.array([1.0,2.0,0.0,0.0,7.52],dtype=jnp.float64)
nu_dp_k = jnp.array([0.0,0.0,1.0,2.0,7.52],dtype=jnp.float64)
nu_k = nu_dp_k - nu_p_k

def domega_dot_drho_actual_deriv(rho, T, Y1, Y2, Y3, Y4):
   
   rateConst  = A * jnp.exp(-KinP[0]/T)
   wmol = (KinP[1]+KinP[2])  * rateConst * (rho**(KinP[1]+KinP[2]-1)) * ((Y1/W_k[0])**KinP[1]) * (((Y2/W_k[1]))**KinP[2])
   return wmol

def domega_dot_dT_actual_deriv(rho, T, Y1, Y2, Y3, Y4):
   
   rateConst  = A * jnp.exp(-KinP[0]/T)
   rateConst = KinP[0]*rateConst/(T**2)
   wmol = rateConst * ((rho*(Y1/W_k[0]))**KinP[1]) * ((rho*(Y2/W_k[1]))**KinP[2])
   
   return wmol

def domega_dot_dY1_actual_deriv(rho, T, Y1, Y2, Y3, Y4, A):
   rateConst  = A * jnp.exp(-KinP[0]/T)
   wmol = rateConst *KinP[1]* ((rho*(1/W_k[0]))**KinP[1]) * ((rho*(Y2/W_k[1]))**KinP[2])
   
   return wmol

def domega_dot_dY2_actual_deriv(rho, T, Y1, Y2, Y3, Y4, A):

   rateConst  = A * jnp.exp(-KinP[0]/T)
   wmol = rateConst * KinP[2] * ((rho*(Y1/W_k[0]))**KinP[1]) * ((rho*(1/W_k[1]))**KinP[2])
   
   return wmol

def C_Y_O2(rho, T, Y_CH4, Y_O2, A):
   """
   Computes the value of the given expression:
   -A * exp(-Ea/(R*T)) * rho^2 * (Y_CH4/W_O2) * (1/W_O2) * sum_k(Delta_h_f_k * W_k * nu_k)
   """
   sum_term = jnp.sum(hf * W_k * nu_k)
   result = -1 * domega_dot_dY2_actual_deriv(rho, T, Y_CH4, Y_O2, 0.0, 0.0, A) * sum_term
   return result

def C_Y_CH4(rho, T, Y_CH4, Y_O2, A):
   """
   Computes the value of the given expression:
   -A * exp(-Ea/(R*T)) * rho^2 * (Y_CH4/W_CH4) * (/W_CH4) * sum_k(Delta_h_f_k * W_k * nu_k)
   """
   sum_term = jnp.sum(hf * W_k * nu_k)
   result = -1 * domega_dot_dY1_actual_deriv(rho, T, Y_CH4, Y_O2, 0.0, 0.0, A) * sum_term
   return result