import jax.numpy as jnp
import jax 
from AUTO_DIFF_PACK import read_util as rfu
from AUTO_DIFF_PACK import write_util as wfu
from AUTO_DIFF_PACK import chem_source_term_functions as cstf
import os

#Compute derivatives
def main():
    read_path = r"docs/FEHydro_P1"
    #read_path = r"../.FEHydro_P1"
    write_path = r"docs/Derivs"
    #write_path = r"../.FEHydro/Auto_diff"
    os.makedirs(write_path, exist_ok=True) 
    
    filename_Qbar = r"./Mean_Qbar.txt"
    with open(filename_Qbar, 'r') as file:
        Q_bar = jnp.array([float(file.readline().strip())], dtype=jnp.float64)


    rho_ref = 0.4237
    T_ref = 800#K
    l_ref = 2e-3#m
    U_ref = 65#m/s
    V_ref = l_ref**3
    Cp_ref = 1100e3 #J/kg-K
 
    rhoM = rfu.read_array_from_file(os.path.join(read_path ,'rhobase.txt'))
    rhoM = rho_ref * rhoM
    TM = rfu.read_array_from_file(os.path.join(read_path ,'Tbase.txt'))
    TM = T_ref * TM
    Y1M = rfu.read_array_from_file(os.path.join(read_path ,'Ybase1.txt'))
    Y2M = rfu.read_array_from_file(os.path.join(read_path ,'Ybase2.txt'))
    Y3M = rfu.read_array_from_file(os.path.join(read_path ,'Ybase3.txt'))
    Y4M = rfu.read_array_from_file(os.path.join(read_path ,'Ybase4.txt'))
    Y5M = 1 - (Y1M + Y2M + Y3M + Y4M)
    species_idx = [1,2,3,4,5] #CH4, O2, CO2, H2O, N2
    omega_dot_k_scaling = (rho_ref*U_ref)/l_ref
    omega_dot_T_scaling = (rho_ref*Cp_ref*T_ref*U_ref)/l_ref

    # Compute domega_dot_k_drho_terms---------------------------------------------------
    domega_dot_CH4_drho = jnp.diag(jax.jacfwd(cstf.omega_dot_CH4,0)(rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M))
    domega_dot_CH4_drho_s = (domega_dot_CH4_drho*rho_ref)/(omega_dot_k_scaling) 
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[0]) + "_drho", domega_dot_CH4_drho_s)
    del domega_dot_CH4_drho_s

    domega_dot_O2_drho = jnp.diag(jax.jacfwd(cstf.omega_dot_O2,0)(rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M))
    domega_dot_O2_drho_s = (domega_dot_O2_drho*rho_ref)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[1] + 1) + "_drho", domega_dot_O2_drho_s)
    del domega_dot_O2_drho_s

    domega_dot_CO2_drho = jnp.diag(jax.jacfwd(cstf.omega_dot_CO2,0)(rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M))
    domega_dot_CO2_drho_s = (domega_dot_CO2_drho*rho_ref)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[2] + 1) + "_drho", domega_dot_CO2_drho_s)
    del domega_dot_CO2_drho_s

    domega_dot_H2O_drho = jnp.diag(jax.jacfwd(cstf.omega_dot_H2O,0)(rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M))
    domega_dot_H2O_drho_s = (domega_dot_H2O_drho*rho_ref)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[3] + 1) + "_drho", domega_dot_H2O_drho_s)
    del domega_dot_H2O_drho_s

    domega_dot_N2_drho = jnp.diag(jax.jacfwd(cstf.omega_dot_N2,0)(rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M))
    domega_dot_N2_drho_s = (domega_dot_N2_drho*rho_ref)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[4]) + "_drho", domega_dot_N2_drho_s)
    del domega_dot_N2_drho_s
    #Compute domega_dot_T_drho---------------------------------------------------------
    domega_dot_T_drho = cstf.HRR_differentials(domega_dot_CH4_drho, domega_dot_O2_drho, domega_dot_CO2_drho, domega_dot_H2O_drho, domega_dot_N2_drho)

    cn_rho = domega_dot_T_drho * rho_ref * V_ref / Q_bar
    domega_dot_T_drho = (domega_dot_T_drho*rho_ref)/(omega_dot_T_scaling)
    
    wfu.write_to_file(write_path, "domega_dot_T_drho", domega_dot_T_drho)
    wfu.write_to_file(write_path, "CN_rho", cn_rho)
    del cn_rho
    del domega_dot_T_drho
    del domega_dot_CH4_drho 
    del domega_dot_O2_drho
    del domega_dot_CO2_drho
    del domega_dot_H2O_drho 
    del domega_dot_N2_drho
    #Compute domega_dot_k_dT_terms---------------------------------------------------
    domega_dot_CH4_dT = jnp.diag(jax.jacfwd(cstf.omega_dot_CH4,1)(rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M))
    domega_dot_CH4_dT_s = (domega_dot_CH4_dT*T_ref)/(omega_dot_k_scaling)  
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[0]) + "_dT", domega_dot_CH4_dT_s)
    del domega_dot_CH4_dT_s
    
    domega_dot_O2_dT = jnp.diag(jax.jacfwd(cstf.omega_dot_O2,1)(rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M))
    domega_dot_O2_dT_s = (domega_dot_O2_dT*T_ref)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[1] + 1) + "_dT", domega_dot_O2_dT_s)
    del domega_dot_O2_dT_s
    
    domega_dot_CO2_dT = jnp.diag(jax.jacfwd(cstf.omega_dot_CO2,1)(rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M))
    domega_dot_CO2_dT_s = (domega_dot_CO2_dT*T_ref)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[2] + 1) + "_dT", domega_dot_CO2_dT_s)
    del domega_dot_CO2_dT_s

    domega_dot_H2O_dT = jnp.diag(jax.jacfwd(cstf.omega_dot_H2O,1)(rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M))
    domega_dot_H2O_dT_s = (domega_dot_H2O_dT*T_ref)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[3] + 1) + "_dT", domega_dot_H2O_dT_s)
    del domega_dot_H2O_dT_s

    domega_dot_N2_dT = jnp.diag(jax.jacfwd(cstf.omega_dot_N2,1)(rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M))
    domega_dot_N2_dT_s = (domega_dot_N2_dT*T_ref)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[4]) + "_dT", domega_dot_N2_dT_s)
    del domega_dot_N2_dT_s
    #Compute domega_dot_T_dT---------------------------------------------------------
    domega_dot_T_dT = cstf.HRR_differentials(domega_dot_CH4_dT, domega_dot_O2_dT, domega_dot_CO2_dT, domega_dot_H2O_dT, domega_dot_N2_dT)

    cn_T = (domega_dot_T_dT * V_ref * T_ref) / Q_bar
    domega_dot_T_dT = (domega_dot_T_dT*T_ref)/(omega_dot_T_scaling)
    
    wfu.write_to_file(write_path, "domega_dot_T_dT", domega_dot_T_dT)
    wfu.write_to_file(write_path, "CN_T", cn_T)
    del cn_T
    del domega_dot_T_dT
    del domega_dot_CH4_dT
    del domega_dot_O2_dT
    del domega_dot_CO2_dT
    del domega_dot_H2O_dT
    del domega_dot_N2_dT
    #Compute domega_dot_k_dY1_terms---------------------------------------------------
    domega_dot_CH4_dY1 = jnp.diag(jax.jacfwd(cstf.omega_dot_CH4,2)(rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M))
    domega_dot_CH4_dY1_s = (domega_dot_CH4_dY1)/(omega_dot_k_scaling)   
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[0]) + "_dY1", domega_dot_CH4_dY1_s)
    del domega_dot_CH4_dY1_s
    
    domega_dot_O2_dY1 = jnp.diag(jax.jacfwd(cstf.omega_dot_O2,2)(rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M))
    domega_dot_O2_dY1_s = (domega_dot_O2_dY1)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[1] + 1) + "_dY1", domega_dot_O2_dY1_s)
    del domega_dot_O2_dY1_s
    
    domega_dot_CO2_dY1 = jnp.diag(jax.jacfwd(cstf.omega_dot_CO2,2)(rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M))
    domega_dot_CO2_dY1_s = (domega_dot_CO2_dY1)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[2] + 1) + "_dY1", domega_dot_CO2_dY1_s)
    del domega_dot_CO2_dY1_s
    
    domega_dot_H2O_dY1 = jnp.diag(jax.jacfwd(cstf.omega_dot_H2O,2)(rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M))
    domega_dot_H2O_dY1_s = (domega_dot_H2O_dY1)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[3] + 1) + "_dY1", domega_dot_H2O_dY1_s)
    del domega_dot_H2O_dY1_s
    
    domega_dot_N2_dY1 = jnp.diag(jax.jacfwd(cstf.omega_dot_N2,2)(rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M))
    domega_dot_N2_dY1_s = (domega_dot_N2_dY1)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[4]) + "_dY1", domega_dot_N2_dY1_s)
    del domega_dot_N2_dY1_s
    #Compute domega_dot_T_dY1---------------------------------------------------------
    domega_dot_T_dY1 = cstf.HRR_differentials(domega_dot_CH4_dY1, domega_dot_O2_dY1, domega_dot_CO2_dY1, domega_dot_H2O_dY1, domega_dot_N2_dY1)

    cn_Y1 = domega_dot_T_dY1 * V_ref / Q_bar
    domega_dot_T_dY1 = (domega_dot_T_dY1)/(omega_dot_T_scaling)
    
    wfu.write_to_file(write_path, "domega_dot_T_dY1", domega_dot_T_dY1)
    wfu.write_to_file(write_path, "CN_Y1", cn_Y1)
    del cn_Y1
    del domega_dot_T_dY1
    del domega_dot_CH4_dY1
    del domega_dot_O2_dY1
    del domega_dot_CO2_dY1
    del domega_dot_H2O_dY1
    del domega_dot_N2_dY1
    #Compute domega_dot_k_dY2_terms---------------------------------------------------
    domega_dot_CH4_dY2 = jnp.diag(jax.jacfwd(cstf.omega_dot_CH4,3)(rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M))
    domega_dot_CH4_dY2_s = (domega_dot_CH4_dY2)/(omega_dot_k_scaling)  
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[0]) + "_dY2", domega_dot_CH4_dY2_s)
    del domega_dot_CH4_dY2_s

    domega_dot_O2_dY2 = jnp.diag(jax.jacfwd(cstf.omega_dot_O2,3)(rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M))
    domega_dot_O2_dY2_s = (domega_dot_O2_dY2)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[1] + 1) + "_dY2", domega_dot_O2_dY2_s)
    del domega_dot_O2_dY2_s
    
    domega_dot_CO2_dY2 = jnp.diag(jax.jacfwd(cstf.omega_dot_CO2,3)(rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M))
    domega_dot_CO2_dY2_s = (domega_dot_CO2_dY2)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[2] + 1) + "_dY2", domega_dot_CO2_dY2_s)
    del domega_dot_CO2_dY2_s
    
    domega_dot_H2O_dY2 = jnp.diag(jax.jacfwd(cstf.omega_dot_H2O,3)(rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M))
    domega_dot_H2O_dY2_s = (domega_dot_H2O_dY2)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[3] + 1) + "_dY2", domega_dot_H2O_dY2_s)
    del domega_dot_H2O_dY2_s

    domega_dot_N2_dY2 = jnp.diag(jax.jacfwd(cstf.omega_dot_N2,3)(rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M))
    domega_dot_N2_dY2_s = (domega_dot_N2_dY2)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[4]) + "_dY2", domega_dot_N2_dY2_s)
    del domega_dot_N2_dY2_s
    #Compute domega_dot_T_dY2---------------------------------------------------------
    domega_dot_T_dY2 = cstf.HRR_differentials(domega_dot_CH4_dY2, domega_dot_O2_dY2, domega_dot_CO2_dY2, domega_dot_H2O_dY2, domega_dot_N2_dY2)
    
    cn_Y2 = domega_dot_T_dY2 * V_ref / Q_bar
    domega_dot_T_dY2 = (domega_dot_T_dY2)/(omega_dot_T_scaling)
    
    wfu.write_to_file(write_path, "domega_dot_T_dY2", domega_dot_T_dY2)
    wfu.write_to_file(write_path, "CN_Y2", cn_Y2)
    del cn_Y2
    del domega_dot_T_dY2
    del domega_dot_CH4_dY2
    del domega_dot_O2_dY2
    del domega_dot_CO2_dY2
    del domega_dot_H2O_dY2
    del domega_dot_N2_dY2
    #Compute domega_dot_k_dY3_terms---------------------------------------------------
    domega_dot_CH4_dY3 = jnp.diag(jax.jacfwd(cstf.omega_dot_CH4,4)(rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M))
    domega_dot_CH4_dY3_s = (domega_dot_CH4_dY3)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[0]) + "_dY3", domega_dot_CH4_dY3_s)
    del domega_dot_CH4_dY3_s

    domega_dot_O2_dY3 = jnp.diag(jax.jacfwd(cstf.omega_dot_O2,4)(rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M))
    domega_dot_O2_dY3_s = (domega_dot_O2_dY3)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[1] + 1) + "_dY3", domega_dot_O2_dY3_s)
    del domega_dot_O2_dY3_s

    domega_dot_CO2_dY3 = jnp.diag(jax.jacfwd(cstf.omega_dot_CO2,4)(rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M))
    domega_dot_CO2_dY3_s = (domega_dot_CO2_dY3)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[2] + 1) + "_dY3", domega_dot_CO2_dY3_s)
    del domega_dot_CO2_dY3_s

    domega_dot_H2O_dY3 = jnp.diag(jax.jacfwd(cstf.omega_dot_H2O,4)(rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M))
    domega_dot_H2O_dY3_s = (domega_dot_H2O_dY3)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[3] + 1) + "_dY3", domega_dot_H2O_dY3_s)
    del domega_dot_H2O_dY3_s

    domega_dot_N2_dY3 = jnp.diag(jax.jacfwd(cstf.omega_dot_N2,4)(rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M))
    domega_dot_N2_dY3_s = (domega_dot_N2_dY3)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[4]) + "_dY3", domega_dot_N2_dY3_s)
    del domega_dot_N2_dY3_s
    #Compute domega_dot_T_dY3---------------------------------------------------------
    domega_dot_T_dY3 = cstf.HRR_differentials(domega_dot_CH4_dY3, domega_dot_O2_dY3, domega_dot_CO2_dY3, domega_dot_H2O_dY3, domega_dot_N2_dY3)
    
    cn_Y3 = domega_dot_T_dY3 * V_ref / Q_bar
    domega_dot_T_dY3 = (domega_dot_T_dY3)/(omega_dot_T_scaling)
    
    wfu.write_to_file(write_path, "domega_dot_T_dY3", domega_dot_T_dY3)
    wfu.write_to_file(write_path, "CN_Y3", cn_Y3)
    del cn_Y3
    del domega_dot_T_dY3
    del domega_dot_CH4_dY3
    del domega_dot_O2_dY3
    del domega_dot_CO2_dY3
    del domega_dot_H2O_dY3
    del domega_dot_N2_dY3
    #Compute domega_dot_k_dY4_terms---------------------------------------------------
    domega_dot_CH4_dY4 = jnp.diag(jax.jacfwd(cstf.omega_dot_CH4,5)(rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M))
    domega_dot_CH4_dY4_s = (domega_dot_CH4_dY4)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[0]) + "_dY4", domega_dot_CH4_dY4_s)
    del domega_dot_CH4_dY4_s

    domega_dot_O2_dY4 = jnp.diag(jax.jacfwd(cstf.omega_dot_O2,5)(rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M))
    domega_dot_O2_dY4_s = (domega_dot_O2_dY4)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[1] + 1) + "_dY4", domega_dot_O2_dY4_s)
    del domega_dot_O2_dY4_s

    domega_dot_CO2_dY4 = jnp.diag(jax.jacfwd(cstf.omega_dot_CO2,5)(rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M))
    domega_dot_CO2_dY4_s = (domega_dot_CO2_dY4)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[2] + 1) + "_dY4", domega_dot_CO2_dY4_s)
    del domega_dot_CO2_dY4_s

    domega_dot_H2O_dY4 = jnp.diag(jax.jacfwd(cstf.omega_dot_H2O,5)(rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M))
    domega_dot_H2O_dY4_s = (domega_dot_H2O_dY4)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[3] + 1) + "_dY4", domega_dot_H2O_dY4_s)
    del domega_dot_H2O_dY4_s

    domega_dot_N2_dY4 = jnp.diag(jax.jacfwd(cstf.omega_dot_N2,5)(rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M))
    domega_dot_N2_dY4_s = (domega_dot_N2_dY4)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[4]) + "_dY4", domega_dot_N2_dY4_s)
    del domega_dot_N2_dY4_s
    #Compute domega_dot_T_dY4---------------------------------------------------------
    domega_dot_T_dY4 = cstf.HRR_differentials(domega_dot_CH4_dY4, domega_dot_O2_dY4, domega_dot_CO2_dY4, domega_dot_H2O_dY4, domega_dot_N2_dY4)
    
    cn_Y4 = domega_dot_T_dY4 * V_ref / Q_bar
    domega_dot_T_dY4 = (domega_dot_T_dY4)/(omega_dot_T_scaling)
    
    wfu.write_to_file(write_path, "domega_dot_T_dY4", domega_dot_T_dY4)
    wfu.write_to_file(write_path, "CN_Y4", cn_Y4)
    del cn_Y4
    del domega_dot_T_dY4
    del domega_dot_CH4_dY4
    del domega_dot_O2_dY4
    del domega_dot_CO2_dY4
    del domega_dot_H2O_dY4
    del domega_dot_N2_dY4
    #Compute domega_dot_k_dY5_terms---------------------------------------------------
    domega_dot_CH4_dY5 = jnp.diag(jax.jacfwd(cstf.omega_dot_CH4,6)(rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M))
    domega_dot_CH4_dY5_s = (domega_dot_CH4_dY5)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[0]) + "_dY5", domega_dot_CH4_dY5_s)
    del domega_dot_CH4_dY5_s

    domega_dot_O2_dY5 = jnp.diag(jax.jacfwd(cstf.omega_dot_O2,6)(rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M))
    domega_dot_O2_dY5_s = (domega_dot_O2_dY5)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[1] + 1) + "_dY5", domega_dot_O2_dY5_s)
    del domega_dot_O2_dY5_s

    domega_dot_CO2_dY5 = jnp.diag(jax.jacfwd(cstf.omega_dot_CO2,6)(rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M))
    domega_dot_CO2_dY5_s = (domega_dot_CO2_dY5)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[2] + 1) + "_dY5", domega_dot_CO2_dY5_s)
    del domega_dot_CO2_dY5_s

    domega_dot_H2O_dY5 = jnp.diag(jax.jacfwd(cstf.omega_dot_H2O,6)(rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M))
    domega_dot_H2O_dY5_s = (domega_dot_H2O_dY5)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[3] + 1) + "_dY5", domega_dot_H2O_dY5_s)
    del domega_dot_H2O_dY5_s

    domega_dot_N2_dY5 = jnp.diag(jax.jacfwd(cstf.omega_dot_N2,6)(rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M))
    domega_dot_N2_dY5_s = (domega_dot_N2_dY5)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[4]) + "_dY5", domega_dot_N2_dY5_s)
    del domega_dot_N2_dY5_s
    #Compute domega_dot_T_dY5---------------------------------------------------------
    domega_dot_T_dY5 = cstf.HRR_differentials(domega_dot_CH4_dY5, domega_dot_O2_dY5, domega_dot_CO2_dY5, domega_dot_H2O_dY5, domega_dot_N2_dY5)
    
    cn_Y5 = domega_dot_T_dY5 * V_ref / Q_bar
    domega_dot_T_dY5 = (domega_dot_T_dY5)/(omega_dot_T_scaling)
    
    wfu.write_to_file(write_path, "domega_dot_T_dY5", domega_dot_T_dY5)
    wfu.write_to_file(write_path, "CN_Y5", cn_Y5)
    del cn_Y5
    del domega_dot_T_dY5
    del domega_dot_CH4_dY5
    del domega_dot_O2_dY5
    del domega_dot_CO2_dY5
    del domega_dot_H2O_dY5
    del domega_dot_N2_dY5

    print("\nEND\n")
if __name__ == "__main__":
        main()