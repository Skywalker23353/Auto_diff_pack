import jax.numpy as jnp
import jax 
from AUTO_DIFF_PACK import read_util as rfu
from AUTO_DIFF_PACK import write_util as wfu
from AUTO_DIFF_PACK import chem_source_term_functions as cstf
from AUTO_DIFF_PACK import reg_least_sq_fit as rlsf
import os

#Compute derivatives
def main():
    read_path = r"docs/FEHydro_P1_v10"
    #read_path = r"../.FEHydro_P1"
    # write_path = r"docs/Derivs_july_2025_jax_vmap"
    write_path = r"../.FEHydro/Baseflow_CN_P1"
    os.makedirs(write_path, exist_ok=True) 
    
    filename_Qbar = r"./Mean_Qbar.txt"
    with open(filename_Qbar, 'r') as file:
        Q_bar = jnp.array([float(file.readline().strip())], dtype=jnp.float64)

    filename_A = r"./pre-exponential_A.txt"
    with open(filename_A, 'r') as file:
        A_arr = jnp.array([float(file.readline().strip())], dtype=jnp.float64)
    # A = rfu.read_array_from_file(os.path.join(write_path ,'pre_exponential_field.txt'))

    Ea_val = 31.588e3 # cal/mol
    Ea_val = Ea_val*4.184 # J/mol

    rho_ref = 0.4237
    T_ref = 800#K
    l_ref = 2e-3#m
    U_ref = 65#m/s
    V_ref = l_ref**3
    Cp_ref = 1100 #J/kg-K

    # Species Data
    h_f1 = -5.421277e06 #J/kg
    h_f2 = 4.949450e05 #J/kg
    h_f3 = -8.956200e06 #J/kg
    h_f4 = -1.367883e07 #J/kg
    h_f5 = 5.370115e05 #J/kg

    W_k_CH4 = 16e-3 #kg/mol
    W_k_O2 = 32e-3 #kg/mol
    W_k_CO2 = 44e-3 #kg/mol
    W_k_H2O = 18e-3 #kg/mol
    W_k_N2 = 28e-3 #kg/mol

    nu_k_CH4 = -1.0
    nu_k_O2 = -2.0
    nu_k_CO2 = 1.0
    nu_k_H2O = 2.0
    nu_k_N2 = 0.0
    # W_k = jnp.array([16e-3,32e-3,44e-3,18e-3,28e-3],dtype=jnp.float64) #kg/mol
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
    Y5M = 1 - (Y1M + Y2M + Y3M + Y4M)

    A = A_arr*jnp.ones(rhoM.shape, dtype=jnp.float64)
    Ea = Ea_val*jnp.ones(rhoM.shape, dtype=jnp.float64)
    
    W_k_CH4_vec = W_k_CH4*jnp.ones(rhoM.shape, dtype=jnp.float64)
    W_k_O2_vec = W_k_O2*jnp.ones(rhoM.shape, dtype=jnp.float64)
    W_k_CO2_vec = W_k_CO2*jnp.ones(rhoM.shape, dtype=jnp.float64)
    W_k_H2O_vec = W_k_H2O*jnp.ones(rhoM.shape, dtype=jnp.float64)
    W_k_N2_vec = W_k_N2*jnp.ones(rhoM.shape, dtype=jnp.float64)
    W_k = (W_k_CH4_vec, W_k_O2_vec, W_k_CO2_vec, W_k_H2O_vec, W_k_N2_vec)

    nu_k_CH4_vec = nu_k_CH4*jnp.ones(rhoM.shape, dtype=jnp.float64)
    nu_k_O2_vec = nu_k_O2*jnp.ones(rhoM.shape, dtype=jnp.float64)
    nu_k_CO2_vec = nu_k_CO2*jnp.ones(rhoM.shape, dtype=jnp.float64)
    nu_k_H2O_vec = nu_k_H2O*jnp.ones(rhoM.shape, dtype=jnp.float64)
    nu_k_N2_vec = nu_k_N2*jnp.ones(rhoM.shape, dtype=jnp.float64)
    nu_k = (nu_k_CH4_vec, nu_k_O2_vec, nu_k_CO2_vec, nu_k_H2O_vec, nu_k_N2_vec)

    h_f1_vec = h_f1*jnp.ones(rhoM.shape, dtype=jnp.float64)
    h_f2_vec = h_f2*jnp.ones(rhoM.shape, dtype=jnp.float64)
    h_f3_vec = h_f3*jnp.ones(rhoM.shape, dtype=jnp.float64)
    h_f4_vec = h_f4*jnp.ones(rhoM.shape, dtype=jnp.float64)
    h_f5_vec = h_f5*jnp.ones(rhoM.shape, dtype=jnp.float64)
    h_f = (h_f1_vec, h_f2_vec, h_f3_vec, h_f4_vec, h_f5_vec)
    
    if not os.path.exists(os.path.join(read_path, 'epsilon.txt')):
        print("Kappa and epsilon files not found. Setting the values to zero. \n")
        kappa = jnp.zeros(rhoM.shape, dtype=jnp.float64)
        epsilon = jnp.zeros(rhoM.shape, dtype=jnp.float64)
    else:
        kappa = rfu.read_array_from_file(os.path.join(read_path ,'TKE.txt')) #Turbulent Kinetic Energy
        epsilon = rfu.read_array_from_file(os.path.join(read_path ,'epsilon.txt')) #Turbulent dissipation rate

    # Read omega_dot_T_LES from file
    omega_dot_T_LES = rfu.read_array_from_file(os.path.join(read_path, 'HRRbase.txt'))
    omega_dot_T_LES_rms = rfu.read_array_from_file(os.path.join(read_path, 'HRRrms.txt'))
    N_samples = 1160
    
    omega_dot_T_vmap = jax.vmap(cstf.omega_dot_T, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    omega_dot_T_model_prior = omega_dot_T_vmap(rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M,
                                         A, Ea, kappa, epsilon, W_k, nu_k, h_f)
    
    rmse_prior = rlsf.compute_rmse(omega_dot_T_LES, omega_dot_T_model_prior)
    nrmse_prior = rlsf.compute_nrmse(omega_dot_T_LES, omega_dot_T_model_prior)
    print(f"RMSE: {rmse_prior:.6e}")
    print(f"Normalized RMSE: {nrmse_prior:.6f}")  # Should be << 1.0

    # Fit A and Ea using regularized least squares
    A_s_opt, Ea_s_opt = rlsf.fit_A_and_Ea(rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M,
                                  A, Ea, W_k, nu_k, h_f, kappa, epsilon,
                                   omega_dot_T_LES, omega_dot_T_LES_rms, N_samples, lambda_reg=0.01)
    
    print("Completed fitting A and Ea.\n")
    print("A_s_opt:", A_s_opt)
    print("Ea_s_opt:", Ea_s_opt)

    # Use optimized A and Ea for final calculations
    A_arr_opt = A_s_opt * A_arr
    Ea_val_opt = Ea_s_opt * Ea_val

    print("A_arr_opt:", A_arr_opt)
    print("Ea_arr_opt:", Ea_val_opt)
    
    del A 
    del Ea

    A = A_arr_opt*jnp.ones(rhoM.shape, dtype=jnp.float64)
    Ea = Ea_val_opt*jnp.ones(rhoM.shape, dtype=jnp.float64)

    omega_dot_T_model_post = omega_dot_T_vmap(rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M,
                                         A, Ea, kappa, epsilon, W_k, nu_k, h_f)
    
    rmse_post = rlsf.compute_rmse(omega_dot_T_LES, omega_dot_T_model_post)
    nrmse_post = rlsf.compute_nrmse(omega_dot_T_LES, omega_dot_T_model_post)
    print(f"RMSE: {rmse_post:.6e}")
    print(f"Normalized RMSE: {nrmse_post:.6f}")

    species_idx = [1,2,3,4,5] #CH4, O2, CO2, H2O, N2
    omega_dot_k_scaling = (rho_ref*U_ref)/l_ref
    omega_dot_T_scaling = (rho_ref*Cp_ref*T_ref*U_ref)/l_ref

    domega_dot_CH4_grad = jax.vmap(jax.grad(cstf.omega_dot_CH4, argnums=(0,1,2,3,4,5,6)))
    domega_dot_O2_grad = jax.vmap(jax.grad(cstf.omega_dot_O2, argnums=(0,1,2,3,4,5,6)))
    domega_dot_CO2_grad = jax.vmap(jax.grad(cstf.omega_dot_CO2, argnums=(0,1,2,3,4,5,6)))
    domega_dot_H20_grad = jax.vmap(jax.grad(cstf.omega_dot_H2O, argnums=(0,1,2,3,4,5,6)))
    domega_dot_N2_grad = jax.vmap(jax.grad(cstf.omega_dot_N2, argnums=(0,1,2,3,4,5,6)))

    domega_dot_CH4_drho, domega_dot_CH4_dT, domega_dot_CH4_dY1, domega_dot_CH4_dY2, domega_dot_CH4_dY3, domega_dot_CH4_dY4, domega_dot_CH4_dY5 = domega_dot_CH4_grad(rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M, A,  Ea, kappa, epsilon, W_k, nu_k)
    domega_dot_O2_drho, domega_dot_O2_dT, domega_dot_O2_dY1, domega_dot_O2_dY2, domega_dot_O2_dY3, domega_dot_O2_dY4, domega_dot_O2_dY5 = domega_dot_O2_grad(rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M, A, Ea, kappa, epsilon, W_k, nu_k)
    domega_dot_CO2_drho, domega_dot_CO2_dT, domega_dot_CO2_dY1, domega_dot_CO2_dY2, domega_dot_CO2_dY3, domega_dot_CO2_dY4, domega_dot_CO2_dY5 = domega_dot_CO2_grad(rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M, A, Ea, kappa, epsilon, W_k, nu_k)
    domega_dot_H2O_drho, domega_dot_H2O_dT, domega_dot_H2O_dY1, domega_dot_H2O_dY2, domega_dot_H2O_dY3, domega_dot_H2O_dY4, domega_dot_H2O_dY5 = domega_dot_H20_grad(rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M, A, Ea, kappa, epsilon, W_k, nu_k)
    domega_dot_N2_drho, domega_dot_N2_dT, domega_dot_N2_dY1, domega_dot_N2_dY2, domega_dot_N2_dY3, domega_dot_N2_dY4, domega_dot_N2_dY5 = domega_dot_N2_grad(rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M, A, Ea, kappa, epsilon, W_k, nu_k)
    domega_dot_T_grad = jax.vmap(jax.grad(cstf.omega_dot_T, argnums=(0,1,2,3,4,5,6)))
    domega_dot_T_drho, domega_dot_T_dT, domega_dot_T_dY1, domega_dot_T_dY2, domega_dot_T_dY3, domega_dot_T_dY4, domega_dot_T_dY5 = domega_dot_T_grad(rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M, A, Ea, kappa, epsilon, W_k, nu_k, h_f)

    # Compute domega_dot_k_drho_terms---------------------------------------------------
    domega_dot_CH4_drho_s = (domega_dot_CH4_drho*rho_ref)/(omega_dot_k_scaling) 
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[0]) + "_drho", domega_dot_CH4_drho_s)
    del domega_dot_CH4_drho_s

    domega_dot_O2_drho_s = (domega_dot_O2_drho*rho_ref)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[1]) + "_drho", domega_dot_O2_drho_s)
    del domega_dot_O2_drho_s

    domega_dot_CO2_drho_s = (domega_dot_CO2_drho*rho_ref)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[2]) + "_drho", domega_dot_CO2_drho_s)
    del domega_dot_CO2_drho_s

    domega_dot_H2O_drho_s = (domega_dot_H2O_drho*rho_ref)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[3]) + "_drho", domega_dot_H2O_drho_s)
    del domega_dot_H2O_drho_s

    domega_dot_N2_drho_s = (domega_dot_N2_drho*rho_ref)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[4]) + "_drho", domega_dot_N2_drho_s)
    del domega_dot_N2_drho_s
    #Compute domega_dot_T_drho---------------------------------------------------------
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
    domega_dot_CH4_dT_s = (domega_dot_CH4_dT*T_ref)/(omega_dot_k_scaling)  
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[0]) + "_dT", domega_dot_CH4_dT_s)
    del domega_dot_CH4_dT_s
    
    domega_dot_O2_dT_s = (domega_dot_O2_dT*T_ref)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[1]) + "_dT", domega_dot_O2_dT_s)
    del domega_dot_O2_dT_s
    
    domega_dot_CO2_dT_s = (domega_dot_CO2_dT*T_ref)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[2]) + "_dT", domega_dot_CO2_dT_s)
    del domega_dot_CO2_dT_s

    domega_dot_H2O_dT_s = (domega_dot_H2O_dT*T_ref)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[3]) + "_dT", domega_dot_H2O_dT_s)
    del domega_dot_H2O_dT_s

    domega_dot_N2_dT_s = (domega_dot_N2_dT*T_ref)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[4]) + "_dT", domega_dot_N2_dT_s)
    del domega_dot_N2_dT_s
    #Compute domega_dot_T_dT---------------------------------------------------------
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
    domega_dot_CH4_dY1_s = (domega_dot_CH4_dY1)/(omega_dot_k_scaling)   
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[0]) + "_dY1", domega_dot_CH4_dY1_s)
    del domega_dot_CH4_dY1_s
    
    domega_dot_O2_dY1_s = (domega_dot_O2_dY1)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[1]) + "_dY1", domega_dot_O2_dY1_s)
    del domega_dot_O2_dY1_s
    
    domega_dot_CO2_dY1_s = (domega_dot_CO2_dY1)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[2]) + "_dY1", domega_dot_CO2_dY1_s)
    del domega_dot_CO2_dY1_s
    
    domega_dot_H2O_dY1_s = (domega_dot_H2O_dY1)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[3]) + "_dY1", domega_dot_H2O_dY1_s)
    del domega_dot_H2O_dY1_s
    
    domega_dot_N2_dY1_s = (domega_dot_N2_dY1)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[4]) + "_dY1", domega_dot_N2_dY1_s)
    del domega_dot_N2_dY1_s
    #Compute domega_dot_T_dY1---------------------------------------------------------
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
    domega_dot_CH4_dY2_s = (domega_dot_CH4_dY2)/(omega_dot_k_scaling)  
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[0]) + "_dY2", domega_dot_CH4_dY2_s)
    del domega_dot_CH4_dY2_s

    domega_dot_O2_dY2_s = (domega_dot_O2_dY2)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[1]) + "_dY2", domega_dot_O2_dY2_s)
    del domega_dot_O2_dY2_s
    
    domega_dot_CO2_dY2_s = (domega_dot_CO2_dY2)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[2]) + "_dY2", domega_dot_CO2_dY2_s)
    del domega_dot_CO2_dY2_s
    
    domega_dot_H2O_dY2_s = (domega_dot_H2O_dY2)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[3]) + "_dY2", domega_dot_H2O_dY2_s)
    del domega_dot_H2O_dY2_s

    domega_dot_N2_dY2_s = (domega_dot_N2_dY2)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[4]) + "_dY2", domega_dot_N2_dY2_s)
    del domega_dot_N2_dY2_s
    #Compute domega_dot_T_dY2---------------------------------------------------------
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
    domega_dot_CH4_dY3_s = (domega_dot_CH4_dY3)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[0]) + "_dY3", domega_dot_CH4_dY3_s)
    del domega_dot_CH4_dY3_s

    domega_dot_O2_dY3_s = (domega_dot_O2_dY3)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[1]) + "_dY3", domega_dot_O2_dY3_s)
    del domega_dot_O2_dY3_s

    domega_dot_CO2_dY3_s = (domega_dot_CO2_dY3)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[2]) + "_dY3", domega_dot_CO2_dY3_s)
    del domega_dot_CO2_dY3_s

    domega_dot_H2O_dY3_s = (domega_dot_H2O_dY3)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[3]) + "_dY3", domega_dot_H2O_dY3_s)
    del domega_dot_H2O_dY3_s

    domega_dot_N2_dY3_s = (domega_dot_N2_dY3)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[4]) + "_dY3", domega_dot_N2_dY3_s)
    del domega_dot_N2_dY3_s
    #Compute domega_dot_T_dY3---------------------------------------------------------
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
    domega_dot_CH4_dY4_s = (domega_dot_CH4_dY4)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[0]) + "_dY4", domega_dot_CH4_dY4_s)
    del domega_dot_CH4_dY4_s

    
    domega_dot_O2_dY4_s = (domega_dot_O2_dY4)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[1]) + "_dY4", domega_dot_O2_dY4_s)
    del domega_dot_O2_dY4_s

    domega_dot_CO2_dY4_s = (domega_dot_CO2_dY4)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[2]) + "_dY4", domega_dot_CO2_dY4_s)
    del domega_dot_CO2_dY4_s

    domega_dot_H2O_dY4_s = (domega_dot_H2O_dY4)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[3]) + "_dY4", domega_dot_H2O_dY4_s)
    del domega_dot_H2O_dY4_s

    domega_dot_N2_dY4_s = (domega_dot_N2_dY4)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[4]) + "_dY4", domega_dot_N2_dY4_s)
    del domega_dot_N2_dY4_s
    #Compute domega_dot_T_dY4---------------------------------------------------------
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
    domega_dot_CH4_dY5_s = (domega_dot_CH4_dY5)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[0]) + "_dY5", domega_dot_CH4_dY5_s)
    del domega_dot_CH4_dY5_s

    domega_dot_O2_dY5_s = (domega_dot_O2_dY5)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[1]) + "_dY5", domega_dot_O2_dY5_s)
    del domega_dot_O2_dY5_s

    domega_dot_CO2_dY5_s = (domega_dot_CO2_dY5)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[2]) + "_dY5", domega_dot_CO2_dY5_s)
    del domega_dot_CO2_dY5_s

    domega_dot_H2O_dY5_s = (domega_dot_H2O_dY5)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[3]) + "_dY5", domega_dot_H2O_dY5_s)
    del domega_dot_H2O_dY5_s

    domega_dot_N2_dY5_s = (domega_dot_N2_dY5)/(omega_dot_k_scaling)
    wfu.write_to_file(write_path, "domega_dot_" + str(species_idx[4]) + "_dY5", domega_dot_N2_dY5_s)
    del domega_dot_N2_dY5_s
    #Compute domega_dot_T_dY5---------------------------------------------------------
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

    print("\nDONE\n")
if __name__ == "__main__":
        main()
