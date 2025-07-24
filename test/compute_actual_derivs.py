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
    Cp_ref = 1100#J/kg-K

    # W_k = jnp.array([16e-3,32e-3,44e-3,18e-3,28e-3],dtype=jnp.float64)
    # nu_p_k = jnp.array([1.0,2.0,0.0,0.0,7.52],dtype=jnp.float64)
    # nu_dp_k = jnp.array([0.0,0.0,1.0,2.0,7.52],dtype=jnp.float64)
    # nu_k = nu_dp_k - nu_p_k
 
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
    omega_dot_T_scaling = (rho_ref*Cp_ref*T_ref*U_ref)/l_ref

    #Species
    # CH4, O2, CO2, H2O
    # Variables 
    # rho, T, Y1, Y2, 
    # Derivatives are zero with respect to Y3, Y4
    # Total variables 6 ; Effective variables 4

    # compute actual derivatives
    # Section 1: Compute omega_dot_CH4 differentials
    # Section 2: Compute omega_dot_O2 differentials
    # Section 3: Compute omega_dot_CO2 differentials
    # Section 4: Compute omega_dot_H2O differentials

    #Section 1: Compute omega_dot_CH4 differentials
    # 1
    # domega_dot_CH4_drho =  nu_k[0]*W_k[0]*fw.domega_dot_drho_actual_deriv(rhoM, TM, Y1M, Y2M, Y3M, Y4M)
    # domega_dot_CH4_drho_s = (domega_dot_CH4_drho*rho_ref)/(omega_dot_k_scaling)
    # wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[0]) + "_drho_actual", domega_dot_CH4_drho_s)
    # del domega_dot_CH4_drho_s

    # #2
    # domega_dot_CH4_dT =  nu_k[0]*W_k[0]*fw.domega_dot_dT_actual_deriv(rhoM, TM, Y1M, Y2M, Y3M, Y4M)
    # domega_dot_CH4_dT_s = (domega_dot_CH4_dT*T_ref)/(omega_dot_k_scaling)
    # wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[0]) + "_dT_actual", domega_dot_CH4_dT_s)
    # del domega_dot_CH4_dT_s

    # #3
    # domega_dot_CH4_dY1 =  nu_k[0]*W_k[0]*fw.domega_dot_dY1_actual_deriv(rhoM, TM, Y1M, Y2M, Y3M, Y4M)
    # domega_dot_CH4_dY1_s = (domega_dot_CH4_dY1)/(omega_dot_k_scaling)
    # wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[0]) + "_dY1_actual", domega_dot_CH4_dY1_s)
    # del domega_dot_CH4_dY1_s

    # #4
    # domega_dot_CH4_dY2 =  nu_k[0]*W_k[0]*fw.domega_dot_dY2_actual_deriv(rhoM, TM, Y1M, Y2M, Y3M, Y4M)
    # domega_dot_CH4_dY2_s = (domega_dot_CH4_dY2)/(omega_dot_k_scaling)
    # wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[0]) + "_dY2_actual", domega_dot_CH4_dY2_s)
    # del domega_dot_CH4_dY2_s

    # #5
    # domega_dot_CH4_dY3 =  0*nu_k[0]*W_k[0]*fw.domega_dot_dY1_actual_deriv(rhoM, TM, Y1M, Y2M, Y3M, Y4M)
    # domega_dot_CH4_dY3_s = (domega_dot_CH4_dY3)/(omega_dot_k_scaling)
    # wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[0]) + "_dY3_actual", domega_dot_CH4_dY3_s)
    # del domega_dot_CH4_dY3_s
    # #6
    # domega_dot_CH4_dY4 =  0*nu_k[0]*W_k[0]*fw.domega_dot_dY1_actual_deriv(rhoM, TM, Y1M, Y2M, Y3M, Y4M)
    # domega_dot_CH4_dY4_s = (domega_dot_CH4_dY4)/(omega_dot_k_scaling)
    # wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[0]) + "_dY4_actual", domega_dot_CH4_dY4_s)
    # del domega_dot_CH4_dY4_s

    CN_CH4 = fw.C_Y_CH4(rhoM, TM, Y1M, Y2M, A)
    CN_CH4_s = (CN_CH4 * l_ref ** 3) / Q_bar

    wfu.write_to_file(write_path, "CN_CH4", CN_CH4_s)

    #Section 2: Compute omega_dot_O2 differentials
    # domega_dot_O2_drho =  nu_k[1]*W_k[1]*fw.domega_dot_drho_actual_deriv(rhoM, TM, Y1M, Y2M, Y3M, Y4M)
    # domega_dot_O2_drho_s = (domega_dot_O2_drho*rho_ref)/(omega_dot_k_scaling)
    # wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[1]) + "_drho_actual", domega_dot_O2_drho_s)
    # del domega_dot_O2_drho_s

    # domega_dot_O2_dT =  nu_k[1]*W_k[1]*fw.domega_dot_dT_actual_deriv(rhoM, TM, Y1M, Y2M, Y3M, Y4M)
    # domega_dot_O2_dT_s = (domega_dot_O2_dT*T_ref)/(omega_dot_k_scaling)
    # wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[1]) + "_dT_actual", domega_dot_O2_dT_s)
    # del domega_dot_O2_dT_s

    # domega_dot_O2_dY1 =  nu_k[1]*W_k[1]*fw.domega_dot_dY1_actual_deriv(rhoM, TM, Y1M, Y2M, Y3M, Y4M)
    # domega_dot_O2_dY1_s = (domega_dot_O2_dY1)/(omega_dot_k_scaling)
    # wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[1]) + "_dY1_actual", domega_dot_O2_dY1_s)
    # del domega_dot_O2_dY1_s

    # domega_dot_O2_dY2 =  nu_k[1]*W_k[1]*fw.domega_dot_dY2_actual_deriv(rhoM, TM, Y1M, Y2M, Y3M, Y4M)
    # domega_dot_O2_dY2_s = (domega_dot_O2_dY2)/(omega_dot_k_scaling)
    # wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[1]) + "_dY2_actual", domega_dot_O2_dY2_s)
    # del domega_dot_O2_dY2_s

    # domega_dot_O2_dY3 =  0*nu_k[1]*W_k[1]*fw.domega_dot_dY1_actual_deriv(rhoM, TM, Y1M, Y2M, Y3M, Y4M)
    # domega_dot_O2_dY3_s = (domega_dot_O2_dY3)/(omega_dot_k_scaling)
    # wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[1]) + "_dY3_actual", domega_dot_O2_dY3_s)
    # del domega_dot_O2_dY3_s

    # domega_dot_O2_dY4 =  0*nu_k[1]*W_k[1]*fw.domega_dot_dY1_actual_deriv(rhoM, TM, Y1M, Y2M, Y3M, Y4M)
    # domega_dot_O2_dY4_s = (domega_dot_O2_dY4)/(omega_dot_k_scaling)
    # wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[1]) + "_dY4_actual", domega_dot_O2_dY4_s)
    # del domega_dot_O2_dY4_s

    CN_O2 = fw.C_Y_O2(rhoM, TM, Y1M, Y2M, A)
    CN_O2_s = (CN_O2 * l_ref ** 3) / Q_bar

    wfu.write_to_file(write_path, "CN_O2", CN_O2_s)

    #Section 3: Compute omega_dot_CO2 differentials
    # domega_dot_CO2_drho =  nu_k[2]*W_k[2]*fw.domega_dot_drho_actual_deriv(rhoM, TM, Y1M, Y2M, Y3M, Y4M)
    # domega_dot_CO2_drho_s = (domega_dot_CO2_drho*rho_ref)/(omega_dot_k_scaling)
    # wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[2]) + "_drho_actual", domega_dot_CO2_drho_s)
    # del domega_dot_CO2_drho_s

    # domega_dot_CO2_dT =  nu_k[2]*W_k[2]*fw.domega_dot_dT_actual_deriv(rhoM, TM, Y1M, Y2M, Y3M, Y4M)
    # domega_dot_CO2_dT_s = (domega_dot_CO2_dT*T_ref)/(omega_dot_k_scaling)
    # wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[2]) + "_dT_actual", domega_dot_CO2_dT_s)
    # del domega_dot_CO2_dT_s

    # domega_dot_CO2_dY1 =  nu_k[2]*W_k[2]*fw.domega_dot_dY1_actual_deriv(rhoM, TM, Y1M, Y2M, Y3M, Y4M)
    # domega_dot_CO2_dY1_s = (domega_dot_CO2_dY1)/(omega_dot_k_scaling)
    # wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[2]) + "_dY1_actual", domega_dot_CO2_dY1_s)
    # del domega_dot_CO2_dY1_s

    # domega_dot_CO2_dY2 =  nu_k[2]*W_k[2]*fw.domega_dot_dY2_actual_deriv(rhoM, TM, Y1M, Y2M, Y3M, Y4M)
    # domega_dot_CO2_dY2_s = (domega_dot_CO2_dY2)/(omega_dot_k_scaling)
    # wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[2]) + "_dY2_actual", domega_dot_CO2_dY2_s)
    # del domega_dot_CO2_dY2_s

    # domega_dot_CO2_dY3 =  0*nu_k[2]*W_k[2]*fw.domega_dot_dY1_actual_deriv(rhoM, TM, Y1M, Y2M, Y3M, Y4M)
    # domega_dot_CO2_dY3_s = (domega_dot_CO2_dY3)/(omega_dot_k_scaling)
    # wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[2]) + "_dY3_actual", domega_dot_CO2_dY3_s)
    # del domega_dot_CO2_dY3_s

    # domega_dot_CO2_dY4 =  0*nu_k[2]*W_k[2]*fw.domega_dot_dY1_actual_deriv(rhoM, TM, Y1M, Y2M, Y3M, Y4M)
    # domega_dot_CO2_dY4_s = (domega_dot_CO2_dY4)/(omega_dot_k_scaling)
    # wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[2]) + "_dY4_actual", domega_dot_CO2_dY4_s)
    # del domega_dot_CO2_dY4_s

    # #Section 4: Compute omega_dot_H2O differentials
    # domega_dot_H2O_drho =  nu_k[3]*W_k[3]*fw.domega_dot_drho_actual_deriv(rhoM, TM, Y1M, Y2M, Y3M, Y4M)
    # domega_dot_H2O_drho_s = (domega_dot_H2O_drho*rho_ref)/(omega_dot_k_scaling)
    # wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[3]) + "_drho_actual", domega_dot_H2O_drho_s)
    # del domega_dot_H2O_drho_s
    
    # domega_dot_H2O_dT =  nu_k[3]*W_k[3]*fw.domega_dot_dT_actual_deriv(rhoM, TM, Y1M, Y2M, Y3M, Y4M)
    # domega_dot_H2O_dT_s = (domega_dot_H2O_dT*T_ref)/(omega_dot_k_scaling)
    # wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[3]) + "_dT_actual", domega_dot_H2O_dT_s)
    # del domega_dot_H2O_dT_s

    # domega_dot_H2O_dY1 =  nu_k[3]*W_k[3]*fw.domega_dot_dY1_actual_deriv(rhoM, TM, Y1M, Y2M, Y3M, Y4M)
    # domega_dot_H2O_dY1_s = (domega_dot_H2O_dY1)/(omega_dot_k_scaling)
    # wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[3]) + "_dY1_actual", domega_dot_H2O_dY1_s)
    # del domega_dot_H2O_dY1_s

    # domega_dot_H2O_dY2 =  nu_k[3]*W_k[3]*fw.domega_dot_dY2_actual_deriv(rhoM, TM, Y1M, Y2M, Y3M, Y4M)
    # domega_dot_H2O_dY2_s = (domega_dot_H2O_dY2)/(omega_dot_k_scaling)
    # wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[3]) + "_dY2_actual", domega_dot_H2O_dY2_s)
    # del domega_dot_H2O_dY2_s

    # domega_dot_H2O_dY3 =  0*nu_k[3]*W_k[3]*fw.domega_dot_dY1_actual_deriv(rhoM, TM, Y1M, Y2M, Y3M, Y4M)
    # domega_dot_H2O_dY3_s = (domega_dot_H2O_dY3)/(omega_dot_k_scaling)
    # wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[3]) + "_dY3_actual", domega_dot_H2O_dY3_s)
    # del domega_dot_H2O_dY3_s

    # domega_dot_H2O_dY4 =  0*nu_k[3]*W_k[3]*fw.domega_dot_dY1_actual_deriv(rhoM, TM, Y1M, Y2M, Y3M, Y4M)
    # domega_dot_H2O_dY4_s = (domega_dot_H2O_dY4)/(omega_dot_k_scaling)
    # wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[3]) + "_dY4_actual", domega_dot_H2O_dY4_s)
    # del domega_dot_H2O_dY4_s
    print("done")

if __name__ == "__main__":
    main()

