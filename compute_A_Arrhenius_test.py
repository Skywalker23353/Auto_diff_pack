import os
import numpy as np
import logging
import sys
from AUTO_DIFF_PACK import compute_volume_integral as cvi
from AUTO_DIFF_PACK import read_h5file_util as rh5
# from AUTO_DIFF_PACK import compute_wt_by_C_EBU as fw_EBU
from AUTO_DIFF_PACK import chem_source_term_functions as fw

# Ensure log directory exists
os.makedirs("logs", exist_ok=True)

# Automatically set log file name based on script name
script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
log_file_path = f"logs/{script_name}.log"

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode="w"),  # Overwrite log file each run
        # logging.StreamHandler()  # Also log to console
    ]
)

R = 8.314 # J/molK
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

def main(): 
    logging.info("Entered main()")
    file_path = r"/work/home/satyam/satyam_files/CH4_jet_PF/2025_Runs/LES_base_case_v6/filtering_run3/Final_baseflow_with_EBU_components" 
    phasename = 'Reactants'
    filename = phasename
    field = 'Heatrelease'
    indx = 1011260202
    
    Y_O2_B = 0.0423
    Y_O2_U = 0.2226
    
    grid_name = 'burner' 
    blks = np.arange(0, 28, 1)
    # field_name = 'Heatrelease'
    # local_heat_release_rate_LES = np.zeros(blks.shape, dtype=np.float64)
    local_heat_release_rate_model = np.zeros(blks.shape, dtype=np.float64)
    logging.info("Starting main computation loop")
    for blk in blks:
        logging.info(f"main(): Processing block {blk}")
        logging.debug("Calling rh5.hdf5_read_LES_data for rho")
        rho = rh5.hdf5_read_LES_data(file_path, filename, indx, phasename, grid_name, blk, 'rho_mean')
        logging.debug("Max of rho: %f", np.max(rho))
        logging.debug("Min of rho: %f", np.min(rho))
        logging.debug("Calling rh5.hdf5_read_LES_data for T")
        T = rh5.hdf5_read_LES_data(file_path, filename, indx, phasename, grid_name, blk, 'T_fmean')
        logging.debug("Max of T: %f", np.max(T))
        logging.debug("Min of T: %f", np.min(T))
        logging.debug("Calling rh5.hdf5_read_LES_data for epsilon")
        # epsilon = rh5.hdf5_read_LES_data(file_path, filename, indx, phasename, grid_name, blk, 'epsilon')
        epsilon = np.zeros(rho.shape, dtype=np.float64)
        # logging.debug("Max of epsilon: %f", np.max(epsilon))
        # logging.debug("Min of epsilon: %f", np.min(epsilon))
        # logging.debug("Calling rh5.hdf5_read_LES_data for kappa")
        # kappa = rh5.hdf5_read_LES_data(file_path, filename, indx, phasename, grid_name, blk, 'TKE')
        kappa = np.zeros(rho.shape, dtype=np.float64)
        # logging.debug("Max of kappa: %f", np.max(kappa))
        # logging.debug("Min of kappa: %f", np.min(kappa))
        logging.debug("Calling rh5.hdf5_read_LES_data for Y_CH4")
        Y_CH4 = rh5.hdf5_read_LES_data(file_path, filename, indx, phasename, grid_name, blk, 'CH4_fmean')
        logging.debug("Max of Y_CH4: %f", np.max(Y_CH4))
        logging.debug("Min of Y_CH4: %f", np.min(Y_CH4))
        logging.debug("Calling rh5.hdf5_read_LES_data for Y_O2")
        Y_O2 = rh5.hdf5_read_LES_data(file_path, filename, indx, phasename, grid_name, blk, 'O2_fmean')
        logging.debug("Max of Y_O2: %f", np.max(Y_O2))
        logging.debug("Min of Y_O2: %f", np.min(Y_O2))
        logging.debug("Calling rh5.hdf5_read_LES_data for Y_CO2")
        Y_CO2 = rh5.hdf5_read_LES_data(file_path, filename, indx, phasename, grid_name, blk, 'CO2_fmean')
        logging.debug("Max of Y_CO2: %f", np.max(Y_CO2))
        logging.debug("Min of Y_CO2: %f", np.min(Y_CO2))
        logging.debug("Calling rh5.hdf5_read_LES_data for Y_H2O")
        Y_H2O = rh5.hdf5_read_LES_data(file_path, filename, indx, phasename, grid_name, blk, 'H2O_fmean')
        logging.debug("Max of Y_H2O: %f", np.max(Y_H2O))
        logging.debug("Min of Y_H2O: %f", np.min(Y_H2O))
        logging.debug("Calling rh5.hdf5_read_LES_data for Y_N2")
        Y_N2 = rh5.hdf5_read_LES_data(file_path, filename, indx, phasename, grid_name, blk, 'N2')
        logging.debug("Max of Y_N2: %f", np.max(Y_N2))
        logging.debug("Min of Y_N2: %f", np.min(Y_N2))

        A = np.ones(rho.shape, dtype=np.float64)
        logging.debug("Max of Arrhenius factor: %f", np.max(A))

        h_f1_vec = h_f1*np.ones(rho.shape, dtype=np.float64)
        h_f2_vec = h_f2*np.ones(rho.shape, dtype=np.float64)
        h_f3_vec = h_f3*np.ones(rho.shape, dtype=np.float64)
        h_f4_vec = h_f4*np.ones(rho.shape, dtype=np.float64)
        h_f5_vec = h_f5*np.ones(rho.shape, dtype=np.float64)
        h_f = (h_f1_vec, h_f2_vec, h_f3_vec, h_f4_vec, h_f5_vec)

        nu_k_CH4_vec = nu_k_CH4*np.ones(rho.shape, dtype=np.float64)
        nu_k_O2_vec = nu_k_O2*np.ones(rho.shape, dtype=np.float64)
        nu_k_CO2_vec = nu_k_CO2*np.ones(rho.shape, dtype=np.float64)
        nu_k_H2O_vec = nu_k_H2O*np.ones(rho.shape, dtype=np.float64)
        nu_k_N2_vec = nu_k_N2*np.ones(rho.shape, dtype=np.float64)
        nu_k = (nu_k_CH4_vec, nu_k_O2_vec, nu_k_CO2_vec, nu_k_H2O_vec, nu_k_N2_vec)

        W_k_CH4_vec = W_k_CH4*np.ones(rho.shape, dtype=np.float64)
        W_k_O2_vec = W_k_O2*np.ones(rho.shape, dtype=np.float64)
        W_k_CO2_vec = W_k_CO2*np.ones(rho.shape, dtype=np.float64)
        W_k_H2O_vec = W_k_H2O*np.ones(rho.shape, dtype=np.float64)
        W_k_N2_vec = W_k_N2*np.ones(rho.shape, dtype=np.float64)
        W_k = (W_k_CH4_vec, W_k_O2_vec, W_k_CO2_vec, W_k_H2O_vec, W_k_N2_vec)  

        # Y_O2_U_vec = Y_O2_U*np.ones(rho.shape, dtype=np.float64)
        # Y_O2_B_vec = Y_O2_B*np.ones(rho.shape, dtype=np.float64)
        

        logging.debug("Calling fw_EBU.omega_dot_T")
        Model_field = fw.omega_dot_T(
            rho, T, Y_CH4, Y_O2, Y_CO2, Y_H2O, Y_N2, A, kappa, epsilon, W_k, nu_k, h_f
        )
        logging.debug("Calling cvi.compute_vol_integral_of_field")
        local_heat_release_rate_model[blk] = cvi.compute_vol_integral_of_field(
            file_path, filename, phasename, grid_name, blk, Model_field, 0
        )
        logging.info(f"main(): Finished block {blk}")
    #global_heat_release_rate_LES = np.sum(local_heat_release_rate_LES)
    global_heat_release_rate_LES = 163.00
    global_heat_release_rate_by_A_model = np.sum(local_heat_release_rate_model)
    A = global_heat_release_rate_LES/global_heat_release_rate_by_A_model
    logging.info(f"Omega_bar (LES) = {global_heat_release_rate_LES}")
    logging.info(f"A = {A}")
    with open("pre-Exp_factor_A.txt", "w") as f:
        f.write("{:g}\n".format(A))
    # with open("Mean_Qbar.txt", "w") as fQ:
    #     fQ.write("{:g}\n".format(global_heat_release_rate_LES))
    logging.info("main(): Finished all blocks")
    logging.info("main(): Exiting main()")

if __name__ == "__main__":
    logging.info("Script started")
    main()  # Call the main function
    logging.info("Script finished")
