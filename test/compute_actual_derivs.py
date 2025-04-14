# This script computes the derivatives of the volumetric heat release rate with respect to the state variables
from AUTO_DIFF_PACK import f_w_functions as fw
from AUTO_DIFF_PACK import read_util as rfu
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
    V_ref = l_ref**3

    W_k = jnp.array([16e-3,32e-3,44e-3,18e-3,28e-3],dtype=jnp.float64)
    nu_p_k = jnp.array([1.0,2.0,0.0,0.0,7.52],dtype=jnp.float64)
    nu_dp_k = jnp.array([0.0,0.0,1.0,2.0,7.52],dtype=jnp.float64)
    nu_k = nu_dp_k - nu_p_k
 
    rhoM = rfu.read_array_from_file(os.path.join(read_path ,'rhobase.txt'))
    rhoM = rho_ref * rhoM
    TM = rfu.read_array_from_file(os.path.join(read_path ,'Tbase.txt'))
    TM = T_ref * TM
    Y0M = rfu.read_array_from_file(os.path.join(read_path ,'Ybase1.txt'))
    Y0M = Y0M
    Y1M = rfu.read_array_from_file(os.path.join(read_path ,'Ybase2.txt'))
    Y1M = Y1M
    Y2M = rfu.read_array_from_file(os.path.join(read_path ,'Ybase3.txt'))
    Y2M = Y2M
    Y3M = rfu.read_array_from_file(os.path.join(read_path ,'Ybase4.txt'))
    Y3M = Y3M
    #Y5M = read_array_from_file(read_path + '/Ybase5.txt')
    #Y5M = Y5M

    Y = [Y0M,Y1M,Y2M,Y3M]
    species_idx = [1,2,3,4] #CH4, O2, CO2, H2O
    q = [2,3,4,5,6,7] # q[0] = rho, q[1] = T, q[2] = Y0, q[3] = Y1, q[4] = Y2, q[5] = Y3
    scalefactor = 1
    eps = 0 
    omega_dot_k_scaling = (rho_ref*U_ref)/l_ref

    # compute actual derivatives
    dwk_drho_sp0 =  nu_k[0]*W_k[0]*fw.domega_dot_drho_actual_deriv(A,rhoM, TM, Y0M, Y1M, Y2M, Y3M)
    dwk_drho_sp0 = (dwk_drho_sp0*rho_ref)/(omega_dot_k_scaling)
    n = int(len(dwk_drho_sp0))
    with open(write_path + "/domega_dot_" + str(species_idx[0]) + "_drho_actual.txt", 'w') as f:
        f.write("%d\n" % n)
    with open(write_path + "/domega_dot_" + str(species_idx[0]) + "_drho_actual.txt", 'a') as f:
        for item in dwk_drho_sp0:
            f.write("%e\n" % item)
    del dwk_drho_sp0

    dwk_dT_sp0 =  nu_k[0]*W_k[0]*fw.domega_dot_dT_actual_deriv(A,rhoM, TM, Y0M, Y1M, Y2M, Y3M)
    dwk_dT_sp0 = (dwk_dT_sp0*T_ref)/(omega_dot_k_scaling)
    n = int(len(dwk_dT_sp0))
    with open(write_path + "/domega_dot_" + str(species_idx[0]) + "_dT_actual.txt", 'w') as f:
        f.write("%d\n" % n)
    with open(write_path + "/domega_dot_" + str(species_idx[0]) + "_dT_actual.txt", 'a') as f:
        for item in dwk_dT_sp0:
            f.write("%e\n" % item)  
    del dwk_dT_sp0

    dwk_dY_sp0 =  nu_k[0]*W_k[0]*fw.domega_dot_dY1_actual_deriv(A,rhoM, TM, Y0M, Y1M, Y2M, Y3M)
    dwk_dY_sp0 = (dwk_dY_sp0)/(omega_dot_k_scaling)
    n = int(len(dwk_dY_sp0))
    with open(write_path + "/domega_dot_" + str(species_idx[0]) + "_dY1_actual.txt", 'w') as f:
        f.write("%d\n" % n)
    with open(write_path + "/domega_dot_" + str(species_idx[0]) + "_dY1_actual.txt", 'a') as f:
        for item in dwk_dY_sp0:
            f.write("%e\n" % item)
    del dwk_dY_sp0

    dwk_dY_sp0 =  nu_k[0]*W_k[0]*fw.domega_dot_dY2_actual_deriv(A,rhoM, TM, Y0M, Y1M, Y2M, Y3M)
    dwk_dY_sp0 = (dwk_dY_sp0)/(omega_dot_k_scaling)
    n = int(len(dwk_dY_sp0))
    with open(write_path + "/domega_dot_" + str(species_idx[0]) + "_dY2_actual.txt", 'w') as f:
        f.write("%d\n" % n)
    with open(write_path + "/domega_dot_" + str(species_idx[0]) + "_dY2_actual.txt", 'a') as f:
        for item in dwk_dY_sp0:
            f.write("%e\n" % item)
    del dwk_dY_sp0

    dwk_dY_sp1 =  nu_k[1]*W_k[1]*fw.domega_dot_dY1_actual_deriv(A,rhoM, TM, Y0M, Y1M, Y2M, Y3M)
    dwk_dY_sp1 = (dwk_dY_sp1)/(omega_dot_k_scaling)
    n = int(len(dwk_dY_sp1))
    with open(write_path + "/domega_dot_" + str(species_idx[1]) + "_dY1_actual.txt", 'w') as f:
        f.write("%d\n" % n)
    with open(write_path + "/domega_dot_" + str(species_idx[1]) + "_dY1_actual.txt", 'a') as f:
        for item in dwk_dY_sp1:
            f.write("%e\n" % item)
    del dwk_dY_sp1
    
    dwk_dY_sp1 =  nu_k[1]*W_k[1]*fw.domega_dot_dY2_actual_deriv(A,rhoM, TM, Y0M, Y1M, Y2M, Y3M)
    dwk_dY_sp1 = (dwk_dY_sp1)/(omega_dot_k_scaling)
    n = int(len(dwk_dY_sp1))
    with open(write_path + "/domega_dot_" + str(species_idx[1]) + "_dY2_actual.txt", 'w') as f:
        f.write("%d\n" % n)
    with open(write_path + "/domega_dot_" + str(species_idx[1]) + "_dY2_actual.txt", 'a') as f:
        for item in dwk_dY_sp1:
            f.write("%e\n" % item)
    del dwk_dY_sp1
    
    print("done")

if __name__ == "__main__":
    main()
    # main()
