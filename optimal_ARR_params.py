import jax.numpy as jnp
import jax 
import logging 
from AUTO_DIFF_PACK.logging_util import get_logger,setup_logging
from AUTO_DIFF_PACK import read_util as rfu
from AUTO_DIFF_PACK import write_util as wfu
from AUTO_DIFF_PACK import reg_least_sq_fit_simple as rlsf
import os

logger = get_logger()

#Compute derivatives
def main(script_directory):
    read_path = r"docs/FEHydro_P1"
    #read_path = r"../.FEHydro_P1"
    write_path = r"docs/Derivs_mod_ARR"
    # write_path = r"../.FEHydro/Baseflow_CN_P1"
    os.makedirs(write_path, exist_ok=True) 
    
    filename_A = r"./pre-exponential_A.txt"
    with open(filename_A, 'r') as file:
        A_arr = jnp.array([float(file.readline().strip())], dtype=jnp.float64)

    Ea_val = 31.588e3 # cal/mol
    Ea_val = Ea_val*4.184 # J/mol

    rho_ref = 1.0
    T_ref = 800#K

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
 
    ### Read base flow fields ########################################################
    rhoM = rfu.read_array_from_file(os.path.join(read_path ,'rhobase.txt'))
    rhoM = rho_ref * rhoM
    TM = rfu.read_array_from_file(os.path.join(read_path ,'Tbase.txt'))
    TM = T_ref * TM
    Y1M = rfu.read_array_from_file(os.path.join(read_path ,'Ybase1.txt'))
    Y2M = rfu.read_array_from_file(os.path.join(read_path ,'Ybase2.txt'))
    Y3M = rfu.read_array_from_file(os.path.join(read_path ,'Ybase3.txt'))
    Y4M = rfu.read_array_from_file(os.path.join(read_path ,'Ybase4.txt'))
    Y5M = 1 - (Y1M + Y2M + Y3M + Y4M)

    # Read omega_dot_T_LES from file
    omega_dot_T_LES = rfu.read_array_from_file(os.path.join(read_path, 'HRRbase.txt'))
    omega_dot_T_LES_rms = rfu.read_array_from_file(os.path.join(read_path, 'HRRrms.txt'))
    N_samples = 1160

    if not (os.path.exists(os.path.join(read_path, 'epsilon.txt'))):
        logger.info("Kappa and epsilon files not found. Setting the values to zero. \n")
        epsilon = jnp.zeros(rhoM.shape, dtype=jnp.float64)
    else:
        epsilon = rfu.read_array_from_file(os.path.join(read_path ,'epsilon.txt')) #Turbulent dissipation rate
    
    if not (os.path.exists(os.path.join(read_path, 'kappa.txt'))):
        logger.info("Kappa file not found. Setting the values to zero. \n")
        kappa = jnp.zeros(rhoM.shape, dtype=jnp.float64)
    else:
        kappa = rfu.read_array_from_file(os.path.join(read_path ,'kappa.txt')) #Turbulent Kinetic Energy

    Xcoord = rfu.read_array_from_file(os.path.join(read_path ,'xcoord.txt'))
    ####################################################################################
    # limit data to x > 0.15 m
    x_limit = 0.15
    indices = jnp.where(Xcoord >= x_limit)
    logger.info("Number of data points after filtering for x >= %.2f m: %d", x_limit, len(indices[0]))
    rhoM = rhoM[indices]
    TM = TM[indices]
    Y1M = Y1M[indices]
    Y2M = Y2M[indices]
    Y3M = Y3M[indices]
    Y4M = Y4M[indices]
    Y5M = Y5M[indices]
    kappa = kappa[indices]
    epsilon = epsilon[indices]
    omega_dot_T_LES = omega_dot_T_LES[indices]
    omega_dot_T_LES_rms = omega_dot_T_LES_rms[indices]
    ################################################################################
    # Prepare W_k, nu_k, h_f as tuples of arrays
    W_k_CH4_vec = W_k_CH4*jnp.ones(rhoM.shape, dtype=jnp.float64)
    W_k_O2_vec = W_k_O2*jnp.ones(rhoM.shape, dtype=jnp.float64)
    W_k_CO2_vec = W_k_CO2*jnp.ones(rhoM.shape, dtype=jnp.float64)
    W_k_H2O_vec = W_k_H2O*jnp.ones(rhoM.shape, dtype=jnp.float64)
    W_k_N2_vec = W_k_N2*jnp.ones(rhoM.shape, dtype=jnp.float64)
    W_k = (W_k_CH4_vec, W_k_O2_vec, W_k_CO2_vec, W_k_H2O_vec, W_k_N2_vec) #tuple of arrays

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
    ################################################################################
    ### Training #####################################################################
    # Fit A and Ea using regularized least squares
    model_uncty = jnp.array(0.01)
    A_s_init = jnp.array(1.0)
    Ea_s_init = jnp.array(1.0)
    init_params = jnp.array([A_s_init, Ea_s_init, model_uncty])
    lambda_reg = jnp.array(10.0)
    
    A_s_opt, Ea_s_opt = rlsf.fit_A_and_Ea(rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M,
                                  A_arr, Ea_val, W_k, nu_k, h_f, kappa, epsilon,
                                   omega_dot_T_LES, omega_dot_T_LES_rms, N_samples, lambda_reg, init_params)
    
    logger.info("Completed fitting A and Ea.\n")
    logger.info("A_s_opt: %.6e", A_s_opt)
    logger.info("Ea_s_opt: %.6e", Ea_s_opt)

    # Use optimized A and Ea for final calculations
    A_arr_opt = A_s_opt * A_arr
    Ea_val_opt = Ea_s_opt * Ea_val

    wfu.write_to_file(os.path.join(script_directory, 'A_optimized.txt'), A_arr_opt)
    wfu.write_to_file(os.path.join(script_directory, 'Ea_optimized.txt'),Ea_val_opt)

    logger.info("\nDONE\n")

if __name__ == "__main__":
    script_directory = os.path.dirname(os.path.abspath(__file__))
    setup_logging(level=logging.DEBUG, script_dir=script_directory)
    try:
        main(script_directory)
        logger.info("Script completed successfully")
    except Exception as e:
        logger.error("An error occurred: %s", str(e), exc_info=True)
        raise
    
