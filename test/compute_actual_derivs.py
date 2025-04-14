# This script computes the derivatives of the volumetric heat release rate with respect to the state variables
from AUTO_DIFF_PACK import f_w_functions as fw
from AUTO_DIFF_PACK import read_util as rfu
from AUTO_DIFF_PACK import write_util as wfu
import os
import jax.numpy as jnp

#Compute derivatives
def main():
    filename_A_Arrhenius = r"pre-exponential_A.txt"
    with open(filename_A_Arrhenius, 'r') as file:
        A = jnp.array([float(file.readline().strip())], dtype=jnp.float64)

    filename_Qbar = r"Mean_Qbar.txt"
    with open(filename_Qbar, 'r') as file:
        Q_bar = jnp.array([float(file.readline().strip())], dtype=jnp.float64)

    read_path = r"docs/FEHydro_P1"
    # read_path = r"../.FEHydro_P1"
    write_path = r"docs/Derivs"
    # write_path = r"../.FEHydro/Baseflow_CN_P1"
    os.makedirs(write_path, exist_ok=True) 
    rho_ref = 0.4237
    T_ref = 800#K
    l_ref = 2e-3#m
    U_ref = 65#m/s

    W_k = jnp.array([16e-3,32e-3,44e-3,18e-3,28e-3],dtype=jnp.float64)
    nu_p_k = jnp.array([1.0,2.0,0.0,0.0,7.52],dtype=jnp.float64)
    nu_dp_k = jnp.array([0.0,0.0,1.0,2.0,7.52],dtype=jnp.float64)
    nu_k = nu_dp_k - nu_p_k
 
    rhoM = rfu.read_array_from_file(os.path.join(read_path ,'rhobase.txt'))
    rhoM = rho_ref * rhoM
    TM = rfu.read_array_from_file(os.path.join(read_path ,'Tbase.txt'))
    TM = T_ref * TM
    Y1M = rfu.read_array_from_file(os.path.join(read_path ,'Ybase1.txt'))
    Y2M = rfu.read_array_from_file(os.path.join(read_path ,'Ybase2.txt'))
    Y3M = rfu.read_array_from_file(os.path.join(read_path ,'Ybase3.txt'))
    Y4M = rfu.read_array_from_file(os.path.join(read_path ,'Ybase4.txt'))
    Y5M = 1 - Y1M - Y2M - Y3M - Y4M
    species_idx = [1,2,3,4] #CH4, O2, CO2, H2O 
    omega_dot_k_scaling = (rho_ref*U_ref)/l_ref

    # compute actual derivatives
    domega_dot_CH4_drho =  nu_k[0]*W_k[0]*fw.domega_dot_drho_actual_deriv(rhoM, TM, Y1M, Y2M, Y3M, Y4M)
    domega_dot_CH4_drho_s = (domega_dot_CH4_drho*rho_ref)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[0]) + "_drho_actual", domega_dot_CH4_drho_s)
    del domega_dot_CH4_drho_s

    domega_dot_CH4_dT =  nu_k[0]*W_k[0]*fw.domega_dot_dT_actual_deriv(rhoM, TM, Y1M, Y2M, Y3M, Y4M)
    domega_dot_CH4_dT_s = (domega_dot_CH4_dT*T_ref)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[0]) + "_dT_actual", domega_dot_CH4_dT_s)
    del domega_dot_CH4_dT_s

    domega_dot_CH4_dY1 =  nu_k[0]*W_k[0]*fw.domega_dot_dY1_actual_deriv(rhoM, TM, Y1M, Y2M, Y3M, Y4M)
    domega_dot_CH4_dY1_s = (domega_dot_CH4_dY1)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[0]) + "_dY1_actual", domega_dot_CH4_dY1_s)
    del domega_dot_CH4_dY1_s

    domega_dot_CH4_dY2 =  nu_k[0]*W_k[0]*fw.domega_dot_dY2_actual_deriv(rhoM, TM, Y1M, Y2M, Y3M, Y4M)
    domega_dot_CH4_dY2_s = (domega_dot_CH4_dY2)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[0]) + "_dY2_actual", domega_dot_CH4_dY2_s)
    del domega_dot_CH4_dY2_s

    print("done")

if __name__ == "__main__":
    main()

