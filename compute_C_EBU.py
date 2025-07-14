import os
import numpy as np
from AUTO_DIFF_PACK import compute_volume_integral as cvi
from AUTO_DIFF_PACK import read_h5file_util as rh5
from AUTO_DIFF_PACK import compute_wt_by_C_EBU as fw_EBU

R = 8.314 # J/molK
#Species Data
h_f1 = -5.421277e06 #J/kg
h_f2 = 4.949450e05 #J/kg
h_f3 = -8.956200e06 #J/kg
h_f4 = -1.367883e07 #J/kg
h_f5 = 5.370115e05 #J/kg
h_f_all = [h_f1, h_f2, h_f3, h_f4, h_f5]
W_k = np.array([16e-3,32e-3,44e-3,18e-3,28e-3],dtype=np.float64)
nu_p_k = np.array([1.0,2.0,0.0,0.0,7.52],dtype=np.float64)
nu_dp_k = np.array([0.0,0.0,1.0,2.0,7.52],dtype=np.float64)
nu_k = nu_dp_k - nu_p_k

def main(): 
    file_path = r"/work/home/satyam/satyam_files/CH4_jet_PF/2025_Runs/" \
    r"LES_base_case_v6/filtering_run3/Final_baseflow_with_EBU_components" 
    phasename = 'Reactants'
    filename = phasename
    field = 'Heatrelease'
    indx = 1011260202

    grid_name = 'burner' 
    blks = np.arange(0, 28, 1)
    # field_name = 'Heatrelease'
    # local_heat_release_rate_LES = np.zeros(blks.shape, dtype=np.float64)
    local_heat_release_rate_model = np.zeros(blks.shape, dtype=np.float64)
    for blk in blks:
        # LES_field = rh5.hdf5_read_LES_data(file_path, filename, indx, phasename, grid_name, blk, field_name)

        rho = rh5.hdf5_read_LES_data(file_path, filename, indx, phasename, grid_name, blk, 'rho_mean')
        T = rh5.hdf5_read_LES_data(file_path, filename, indx, phasename, grid_name, blk, 'T_fmean')
        epsilon = rh5.hdf5_read_LES_data(file_path, filename, indx, phasename, grid_name, blk, 'epsilon')
        kappa = rh5.hdf5_read_LES_data(file_path, filename, indx, phasename, grid_name, blk, 'TKE')
        Y_CH4 = rh5.hdf5_read_LES_data(file_path, filename, indx, phasename, grid_name, blk, 'CH4_fmean')
        Y_O2 = rh5.hdf5_read_LES_data(file_path, filename, indx, phasename, grid_name, blk, 'O2_fmean')
        Y_CO2 = rh5.hdf5_read_LES_data(file_path, filename, indx, phasename, grid_name, blk, 'CO2_fmean')
        Y_H2O = rh5.hdf5_read_LES_data(file_path, filename, indx, phasename, grid_name, blk, 'H2O_fmean')

        Y_all = [Y_CH4,Y_O2,Y_CO2,Y_H2O]
        
        Model_field = fw_EBU.wT_by_C_EBU(rho, h_f_all, W_k, nu_k,Y_all, epsilon, kappa)
        local_heat_release_rate_model[blk] = cvi.compute_vol_integral_of_field(file_path, filename, phasename, grid_name, blk, Model_field,0)
        # local_heat_release_rate_LES[blk] = cvi.compute_vol_integral_of_field(file_path, filename, phasename, grid_name, blk, LES_field,1)
    #global_heat_release_rate_LES = np.sum(local_heat_release_rate_LES)
    global_heat_release_rate_LES = 163.00
    global_heat_release_rate_by_A_model = np.sum(local_heat_release_rate_model)
    A = global_heat_release_rate_LES/global_heat_release_rate_by_A_model
    print("Omega_bar (LES) = ", global_heat_release_rate_LES)
    print("A = ",A)
    with open("C_EBU.txt", "w") as f:
        f.write("{:g}\n".format(A))
    # with open("Mean_Qbar.txt", "w") as fQ:
    #     fQ.write("{:g}\n".format(global_heat_release_rate_LES))

if __name__ == "__main__":
    main()  # Call the main function
