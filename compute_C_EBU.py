import os
import numpy as np
import logging  # Step 1: Import logging
from AUTO_DIFF_PACK import read_util as rfu
from AUTO_DIFF_PACK import write_util as wfu
from AUTO_DIFF_PACK import chem_source_term_functions_EBU as fw_EBU

# Ensure log directory exists
os.makedirs("logs", exist_ok=True)

# Step 2: Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/compute_C_EBU.log", mode="w"),  # Overwrite log file each run
        # logging.StreamHandler()  # Also log to console
    ]
)

def main(): 
    logging.info("Entered main()")
    read_path = r"../.FEHydro_P1" 
    write_path = r"../.FEHydro/Baseflow_CN_P1"
    os.makedirs(write_path, exist_ok=True)

    R = 8.314 # J/molK
    rho_ref = 0.4237
    T_ref = 800#K
    #Species Data
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

    Y_O2_B = 0.0423
    Y_O2_U = 0.2300
    
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
 
    rhoM = rfu.read_array_from_file_numpy(os.path.join(read_path ,'rhobase.txt'))
    rhoM = rho_ref * rhoM
    TM = rfu.read_array_from_file_numpy(os.path.join(read_path ,'Tbase.txt'))
    TM = T_ref * TM
    Y1M = rfu.read_array_from_file_numpy(os.path.join(read_path ,'Ybase1.txt'))
    Y2M = rfu.read_array_from_file_numpy(os.path.join(read_path ,'Ybase2.txt'))
    Y3M = rfu.read_array_from_file_numpy(os.path.join(read_path ,'Ybase3.txt'))
    Y4M = rfu.read_array_from_file_numpy(os.path.join(read_path ,'Ybase4.txt'))
    Y5M = 1 - (Y1M + Y2M + Y3M + Y4M)

    HRR_M = rfu.read_array_from_file_numpy(os.path.join(read_path ,'HRRbase.txt'))
   
   
    W_k_CH4_vec = W_k_CH4*np.ones(rhoM.shape, dtype=np.float64)
    W_k_O2_vec = W_k_O2*np.ones(rhoM.shape, dtype=np.float64)
    W_k_CO2_vec = W_k_CO2*np.ones(rhoM.shape, dtype=np.float64)
    W_k_H2O_vec = W_k_H2O*np.ones(rhoM.shape, dtype=np.float64)
    W_k_N2_vec = W_k_N2*np.ones(rhoM.shape, dtype=np.float64)
    W_k = (W_k_CH4_vec, W_k_O2_vec, W_k_CO2_vec, W_k_H2O_vec, W_k_N2_vec)

    nu_k_CH4_vec = nu_k_CH4*np.ones(rhoM.shape, dtype=np.float64)
    nu_k_O2_vec = nu_k_O2*np.ones(rhoM.shape, dtype=np.float64)
    nu_k_CO2_vec = nu_k_CO2*np.ones(rhoM.shape, dtype=np.float64)
    nu_k_H2O_vec = nu_k_H2O*np.ones(rhoM.shape, dtype=np.float64)
    nu_k_N2_vec = nu_k_N2*np.ones(rhoM.shape, dtype=np.float64)
    nu_k = (nu_k_CH4_vec, nu_k_O2_vec, nu_k_CO2_vec, nu_k_H2O_vec, nu_k_N2_vec)

    h_f1_vec = h_f1*np.ones(rhoM.shape, dtype=np.float64)
    h_f2_vec = h_f2*np.ones(rhoM.shape, dtype=np.float64)
    h_f3_vec = h_f3*np.ones(rhoM.shape, dtype=np.float64)
    h_f4_vec = h_f4*np.ones(rhoM.shape, dtype=np.float64)
    h_f5_vec = h_f5*np.ones(rhoM.shape, dtype=np.float64)
    h_f = (h_f1_vec, h_f2_vec, h_f3_vec, h_f4_vec, h_f5_vec)

    TEMP = np.ones(rhoM.shape, dtype=np.float64)
    Y_O2_U_vec = Y_O2_U*np.ones(rhoM.shape, dtype=np.float64)
    Y_O2_B_vec = Y_O2_B*np.ones(rhoM.shape, dtype=np.float64)

    #kappa = rfu.read_array_from_file_numpy(os.path.join(read_path ,'TKE.txt')) #Turbulent Kinetic Energy
    kappa = np.ones(rhoM.shape, dtype=np.float64) #Turbulent Kinetic Energy
    #epsilon = rfu.read_array_from_file_numpy(os.path.join(read_path ,'epsilon.txt')) #Turbulent dissipation rate
    epsilon = np.ones(rhoM.shape, dtype=np.float64) #Turbulent dissipation rate
    

    logging.debug("Calling fw_EBU.omega_dot_T")
    Model_field = fw_EBU.omega_dot_T(
        rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M, TEMP, kappa, epsilon, W_k, nu_k, h_f, Y_O2_U_vec, Y_O2_B_vec
    )
    Model_field = Model_field + 1e-8#np.finfo(float).eps  # Avoid division by zero
    C_EBU = HRR_M / Model_field

    wfu.write_to_file(write_path, "C_EBU", C_EBU)
    wfu.write_to_file(write_path, "HRR_by_C_EBU", Model_field)
    logging.info("main(): Finished all blocks")
    logging.info("main(): Exiting main()")

if __name__ == "__main__":
    logging.info("Script started")
    main()  # Call the main function
    logging.info("Script finished")
